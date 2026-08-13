# Reproducing the vLLM-nightly MLA compile failure (Kimi-K2.5 on MI355X)

## TL;DR

`vllm/vllm-openai-rocm:nightly` (2026-08-12) cannot serve `amd/Kimi-K2.5-MXFP4`
on MI355X (gfx950). The aiter Gluon MLA decode kernel fails to compile, all TP
workers die, and the engine never initializes.

```
File "/aiter/ops/triton/gluon/mla_gluon.py", line 1029, in mla_gluon
  gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe,
      mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD < BLOCK_H else None)
expected offsets type layout to be BlockedLayout or SliceLayout
-> Worker proc VllmWorker-7 died unexpectedly (exit code: None)
-> RuntimeError: Engine core initialization failed
```

| Image | Result |
|---|---|
| `vllm/vllm-openai-rocm:v0.22.0` | works (2239.96 tok/s/GPU) |
| `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | works (2236.69 tok/s/GPU) |
| `vllm/vllm-openai-rocm:nightly` | **fails to start** |

## One-command reproducer

```bash
# 1. get the weights (~559 GB) — any fast local disk; /dev/shm used here
huggingface-cli download amd/Kimi-K2.5-MXFP4 --local-dir /dev/shm/model/Kimi-K2.5-MXFP4

# 2. reproduce the failure
./repro_bug.sh

# 3. control run on a known-good image (should start fine)
IMAGE=vllm/vllm-openai-rocm:v0.22.0 ./repro_bug.sh
```

`repro_bug.sh` just runs `vllm serve` on the model with `--tensor-parallel-size 8`
and greps the log for the error. Knobs: `IMAGE`, `MODEL_DIR`, `TP`, `PORT`,
`ROCR_VISIBLE_DEVICES`.

## Environment

- 8x AMD Instinct MI355X (gfx950), ROCm 7.x host driver
- Model: `amd/Kimi-K2.5-MXFP4`, 64 safetensors shards, 559.0 GB (verified complete)
- TP8, `--max-model-len 9416`

## Captured evidence

- `FAILURE_mla_gluon.txt` — trimmed traceback
- `runs/kimi_fp4_8k1k_c128_new/docker.log` — full container log (1121 lines,
  10 occurrences of the layout error)
- `runs/kimi_fp4_8k1k_c128_new/server.log` — vLLM server log
- `runs/kimi_fp4_8k1k_c128_new/env.repro` — exact config used

## Caveat

Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when
reproducing: it independently breaks `custom_all_reduce`'s
`hipIpcGetMemHandle` (aiter issue #4174) and masks this bug behind a different
failure.
