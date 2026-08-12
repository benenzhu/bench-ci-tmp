# Kimi-K2.5 nightly check — upstream vLLM nightly FAILS TO START

| Image | tok/s/GPU | Result |
|---|---:|---|
| baseline `vllm/vllm-openai-rocm:v0.22.0` | 2239.96 | ok |
| `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | **2236.69** | ok (-0.1%) |
| upstream `vllm/vllm-openai-rocm:nightly` | — | **server fails to start** |

All at TP8 / 8k1k / CONC=128 / 384 prompts.

## The upstream nightly failure

`vllm/vllm-openai-rocm:nightly` (2026-08-12) cannot serve Kimi-K2.5-MXFP4 on
MI355X: the aiter Gluon MLA decode kernel fails to compile, all 8 workers die,
and the engine never initializes.

```
File "/aiter/ops/triton/gluon/mla_gluon.py", line 1029, in mla
  gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe,
      mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD < BLOCK_H else None)
expected offsets type layout to be BlockedLayout or SliceLayout
-> Worker proc VllmWorker-7 died unexpectedly
-> RuntimeError: Engine core initialization failed
```

10 occurrences in the log; full excerpt in `FAILURE_mla_gluon.txt`.

Model weights verified complete (64/64 shards, 559.0G), and the same config
works on both the v0.22.0 baseline and the `rocm/atom-dev` repackaged vLLM
v0.25.1 — so this is a genuine regression in the upstream nightly, not a setup
problem. **The ATOM-packaged vLLM remains a working path.**
