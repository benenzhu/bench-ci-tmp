# GLM-5-FP8 on ATOM built for ROCm 10.0.0rc2 — 8k1k c8, TP8

Image: `atom-rocm10:local`, built here from `atom_on_vllm_rocm10.dockerfile`
(see `README.md`). Baseline: 344.61 tok/s/GPU
(`rocm/atom:rocm7.2.2...atom0.1.2.post`).

**Result: does not run. No throughput number.**

Weights load fine (all 8 ranks, ~30 s), then every rank dies during warmup:

```
File "/app/ATOM/atom/model_engine/model_runner.py", line 1226, in warmup_model
    self.forward(dummy_batch)
...
File "/app/ATOM/atom/utils/backends.py", line 649, in __call__
    assert not self._called, "VllmBackend can only be called once"
torch._dynamo.exc.BackendCompilerFailed: AssertionError: VllmBackend can only be called once
```

## Root cause: Dynamo can't trace aiter's pybind enum under torch 2.12

`VllmBackend` is single-shot by construction, so a second call means Dynamo
produced a **second graph** for the model forward. Re-running with
`TORCH_LOGS=graph_breaks,recompiles TORCHDYNAMO_VERBOSE=1` names the culprit
exactly:

```
Graph break in user code at /app/ATOM/atom/model_ops/linear.py:866
Graph Break Reason: Encountered graph break when attempting to trace CALL_KW
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.
  Developer debug context: aiter.jit.module_aiter_core.QuantType
*** While handling this graph break, another graph break occurred: ***
graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop.
  Skipping the frame and falling back to eager, as graph breaks in loops are
  not supported.
```

The chain:

1. `LinearBase.forward` (`atom/model_ops/linear.py:866`) branches on
   `self.quant_type.value == QuantType.No.value`, where `QuantType` is a
   **pybind11 enum** exported by `aiter.jit.module_aiter_core`.
2. torch 2.12's Dynamo cannot trace through that pybind enum and breaks on the
   `CALL_KW` in `mark_trace`'s `wrapped(*args, **kwargs)` forwarding
   (`atom/utils/decorators.py:211`). PyTorch itself flags this as
   *"likely to be a Dynamo bug"*.
3. The break happens **inside a loop**, which Dynamo cannot split, so it skips
   the frame and falls back to eager — producing a second entry into
   `VllmBackend`, which trips the assert.

**Prime suspect is the torch version.** Official ATOM images ship torch 2.13
(`rocm/atom-dev:nightly_*`) or 2.10 (`rocm/atom:*`); ours has **2.12** purely
because the only ROCm 10 base available is a vLLM image. ATOM `dev272` is
evidently developed against 2.13, and 2.12's Dynamo behaves differently here.

## Ruled out: the vLLM install in our image

Our image is the only ATOM image that also contains vLLM (it is built on a vLLM
base), which made "the vLLM install is interfering" a natural suspect. It is
not:

- **`VllmBackend` is ATOM's own class** (`atom/utils/backends.py:478`), forked
  from the vLLM project — the file header carries
  `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`. The name
  is inherited, not an indication that vLLM is called.
- **ATOM server mode never imports real vLLM.** Its `import vllm` statements all
  live under `atom/plugin/vllm/` (plugin mode); we run
  `atom.entrypoints.openai_server`, and the log has no `is_plugin_mode` activity.
- **The crash log contains 0 frames from `site-packages/vllm/`** — every frame is
  `/app/ATOM/atom/...` or `torch/_dynamo/...`.

`atom_novllm.dockerfile` here builds the same image with `pip uninstall -y vllm`
(ATOM still imports fine afterwards) if you want to confirm this empirically,
but the evidence above says it will not change the outcome.

## Also ruled out: `--enforce-eager`

The obvious-looking bypass does not work. `enforce_eager` only gates CUDA graph
capture and is never read anywhere near `compilation_config`. The switch that
actually controls `torch.compile` is **`--level`**
(`atom/model_engine/arg_utils.py:195`, default **3** = `PIECEWISE`), which feeds
`compilation_config.level`; the decorator skips compilation only when that is
`NO_COMPILATION`/`DYNAMO_AS_IS` (`atom/utils/decorators.py:485`).

Note `atom/plugin/config.py:560` claims the decorator "checks enforce_eager to
skip" — that comment is wrong for server mode.

`--level 0` would bypass compilation, but it also disables CUDA graphs, so any
number it produced would be a floor, not something comparable to the 344.61
baseline. Not run for that reason.

## What would fix it

In rough order of preference:

1. **Publish a ROCm 10 ATOM base with torch 2.13** — then ATOM runs on the torch
   it targets and none of this arises.
2. **Make `linear.py:866` not branch on a pybind enum inside the traced region** —
   e.g. hoist `quant_type` to a plain Python int/bool at `__init__` time so
   Dynamo sees a constant. This is the narrow fix and is ATOM-side.
3. Report the pybind-enum tracing failure upstream to PyTorch — Dynamo's own
   hint says it is likely a Dynamo bug.

Artifacts: `logs/glm5_atom/` (first run) and `logs/glm5_atom_diag/`
(with graph-break diagnostics). Reproducer: `repro_glm5_atom_rocm10.sh`.
