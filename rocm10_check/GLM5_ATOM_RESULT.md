# GLM-5-FP8 on ATOM for ROCm 10 — 8k1k c8, TP8

**Result: 350.29 tok/s/GPU, +1.6% vs the 344.61 ROCm 7.2 baseline — no
regression.** Getting there took rebuilding the image properly; jump to
[Resolution](#resolution-rebuild-the-image-properly--glm-5-runs-16) for the
final numbers and the two dockerfiles.

The rest of this document is the diagnosis of why the *first* image failed,
kept because both failures are instructive and one of them (the Dynamo/pybind
issue) is still worth upstreaming.

---

## First attempt: `atom-rocm10:local` (ATOM layered onto a vLLM ROCm 10 image)

Built from `atom_on_vllm_rocm10.dockerfile`. Baseline for comparison:
344.61 tok/s/GPU (`rocm/atom:rocm7.2.2...atom0.1.2.post`).

**Result: did not run.**

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

## The Dynamo fix works — and uncovers a second, independent blocker

`atom_quanttype_dynamo.patch` (in this directory) implements the narrow fix:
mirror the pybind enum into a plain int and have the traced `forward` compare
ints only. A property keeps the mirror in sync, because the online-quantization
path reassigns `quant_type` at `linear.py:687`:

```python
_QT_No = int(QuantType.No.value)          # module-level, constant-foldable
...
@property
def quant_type(self): return self._quant_type
@quant_type.setter
def quant_type(self, value):
    self._quant_type = value
    self._quant_type_value = int(value.value)
...
if self._quant_type_value == _QT_No:      # was: self.quant_type.value == QuantType.No.value
```

7 comparisons in `LinearBase.forward` rewritten; `LinearBase` is the base of
every Linear so one edit covers all of them. No image rebuild needed — ATOM is
`pip install -e`, so the reproducer bind-mounts the patch and `git apply`s it at
container start (`PATCH=... ./repro_glm5_atom_rocm10.sh`).

**Result: the compile failure is gone.** No `VllmBackend can only be called
once`, no graph breaks at all, and startup progresses past compilation into
CUDA graph capture. Then it dies there instead:

```
[AITER] /app/aiter/aiter_meta/csrc/kernels/cache_kernels.cu:4043
        norm_weight dtype must match q dtype
Capturing bs=512, max_q_len=1: 0%|  | 0/11
AsyncIOProcManager(ModelRunner): [ModelRunner1/8] proc died unexpectedly (exitcode=-6)
```

This is an abort inside an aiter C++ cache kernel, unrelated to torch.compile.
It is **not** caused by the fp8 KV cache: re-running with `KVDTYPE=bf16`
produces the identical abort.

The likely cause is again a version mismatch, this time in aiter rather than
torch. Official ATOM images build aiter from source (reporting version `0.0.0`);
ours installs the released wheel **0.1.19** that comes with the vLLM ROCm 10
base. `--kv_cache_dtype fp8` is exactly what the ROCm 7 ATOM baseline uses and
works there, so the kernel contract changed between those aiter builds.

## Resolution: rebuild the image properly — GLM-5 runs, +1.6%

Both blockers were artefacts of the improvised image, and both vanish once ATOM
is built the way the official images are. `rocm101_torch_base.dockerfile` here
builds a ROCm 10.1 + torch base on plain `ubuntu:24.04` (ROCm/torch install
lifted from the Primus training Dockerfile, trimmed to the SDK + torch and to
gfx950 only), and `atom_release_rocm101.dockerfile` is ATOM's own release
dockerfile on top of it, building RCCL and aiter **from source**.

Key deviation from the Primus file: **torch 2.13, not the 2.12 it pins.** The
nightly index carries 2.13 for the same ROCm 10.1 date, and 2.13 is what
official ATOM images ship — 2.12 is precisely what breaks Dynamo.

| | broken image | rebuilt image |
|---|---|---|
| base | vLLM ROCm 10 image | `ubuntu:24.04` |
| ROCm | 10.0.0rc2 | **10.1.0a20260814** |
| torch | 2.12 | **2.13.0** |
| aiter | wheel 0.1.19 | **source-built (`0.0.0`)** |
| size | 61.5 GB | 30.1 GB |

Result on `smci355-ccs-aus-n02-09`, TP8 / 8192x1024 / CONC=8 / 24 prompts:

| | ROCm 7.2 ATOM (baseline) | ROCm 10.1 (rebuilt) | delta |
|---|---:|---:|---:|
| **tok/s/GPU** | **344.61** | **350.29** | **+1.6%** |
| Total throughput | — | 2802.30 tok/s | |
| Mean TTFT | — | 843.00 ms | |
| Mean TPOT | — | 24.74 ms | |
| Successful requests | — | 24 / 24 | |

`VllmBackend can only be called once`: **0 occurrences**.
`norm_weight dtype must match q dtype`: **0 occurrences**.

Note this run did **not** apply `atom_quanttype_dynamo.patch` — confirming that
patch was only ever a torch-2.12 workaround. On 2.13 Dynamo traces the pybind
enum fine and unmodified ATOM source works.

**Verdict: no regression. ROCm 10.1 is +1.6% on GLM-5**, consistent with
DeepSeek-R1 (+4.9%) and Kimi (+10.6%).

## What would fix it

(Kept for the record — this was written before the rebuild above resolved it.)

1. ~~Publish a ROCm 10 ATOM base built the way the official images are~~ — done
   locally; see the two dockerfiles in this directory.
2. **Upstream the Dynamo fix anyway** (`atom_quanttype_dynamo.patch`) — no longer
   needed on 2.13, but it is small and behaviour-preserving, and would make ATOM
   robust to any torch whose Dynamo cannot trace pybind enums.
3. Report the pybind-enum tracing failure to PyTorch — Dynamo's own hint says it
   is likely a Dynamo bug.

Artifacts: `logs/glm5_atom/` (first run), `logs/glm5_atom_diag/` (graph-break
diagnostics), `logs/glm5_atom_patched/` (with the fix applied, showing the aiter
abort). Reproducer: `repro_glm5_atom_rocm10.sh`.
