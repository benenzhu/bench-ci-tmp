# GLM-5-FP8 on ATOM built for ROCm 10.0.0rc2 — 8k1k c8, TP8

Image: `atom-rocm10:local`, built here from
`/dev/shm/ATOM/docker/atom_on_vllm_rocm10.dockerfile` (see `ATOM_IMAGE.md`).
Baseline: 344.61 tok/s/GPU (`rocm/atom:rocm7.2.2...atom0.1.2.post`).

**Result: does not run. No throughput number.**

ATOM loads the model and reaches warmup, then dies in all 8 ranks during the
`torch.compile` of the model forward:

```
File "/app/ATOM/atom/model_engine/model_runner.py", line 1226, in warmup_model
    self.forward(dummy_batch)
...
File "/app/ATOM/atom/utils/backends.py", line 649, in __call__
    assert not self._called, "VllmBackend can only be called once"
torch._dynamo.exc.BackendCompilerFailed: backend='VllmBackend' raised:
AssertionError: VllmBackend can only be called once
```

`VllmBackend` is single-shot by construction; Dynamo invoking the backend a
second time means it produced **more than one graph** for
`atom/models/deepseek_v2.py` `forward` — i.e. a graph break that does not occur
on the torch version ATOM is developed against. This image ships
torch 2.12.0+rocm10.0.0rc2 / Python 3.14, well ahead of ATOM's pin, so the most
likely cause is a Dynamo tracing behaviour change rather than anything
GLM-5-specific.

Not chased further: the fix belongs in ATOM (either tolerate multiple backend
calls or eliminate the graph break), not in the benchmark harness.

Artifacts: `glm5_nightly_check/runs/glm5_fp8_8k1k_c8_new/`
(`docker.log`, `server.log`).
