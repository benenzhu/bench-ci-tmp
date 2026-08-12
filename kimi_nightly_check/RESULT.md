# Kimi-K2.5 nightly check — FAILED TO START (nightly regression)

`vllm/vllm-openai-rocm:nightly` (2026-08-12) **cannot serve Kimi-K2.5-MXFP4 at
all** on MI355X: the aiter Gluon MLA decode kernel fails to compile, all 8
workers die, and the engine never initializes.

```
File "/aiter/ops/triton/gluon/mla_gluon.py", line 1029, in mla
  gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe,
      mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD < BLOCK_H else None)
expected offsets type layout to be BlockedLayout or SliceLayout
-> Worker proc VllmWorker-7 died unexpectedly
-> RuntimeError: Engine core initialization failed
```

10 occurrences in the log; full excerpt in `FAILURE_mla_gluon.txt`.

| Image | Result |
|---|---|
| baseline `vllm/vllm-openai-rocm:v0.22.0` | 2239.96 tok/s/GPU (TP8, 8k1k c128) |
| nightly `vllm/vllm-openai-rocm:nightly` | **server fails to start** |

Model weights verified complete (64/64 shards, 559.0G) and the same config
worked on the baseline image, so this is a genuine nightly regression in the
CDNA4 MLA path, not a setup problem.

`_atomdev_rebuild_runs/` holds an earlier run against the `rocm/atom-dev`
repackaged vLLM v0.25.1, which *did* work and scored 2236.69 (-0.1%) — so the
breakage is specific to the newer upstream nightly.
