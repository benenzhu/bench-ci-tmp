# DeepSeek-R1-0528 on ROCm 10.0.0rc2 — 8k1k c128, TP4 (SGLang)

| Image | ROCm | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---|---:|---:|---:|---:|
| baseline `lmsysorg/sglang-rocm:v0.5.13-rocm720` | 7.2.0 | 4519.73 | — | — | — |
| `lmsysorg/sglang-rocm:v0.5.17-rocm700-mi35x-20260811` | 7.0.0 | 4591.87 | 18367.5 | 4650.7 ms | 56.00 ms |
| **`rocm/ufb-private:sglang-v0.5.15.post1-...rocm10.0.0rc2`** | **10.0.0rc2** | **4742.94** | 18971.8 | 4713.8 ms | 53.96 ms |

vs baseline: **+223.21 (+4.9%) — NEUTRAL, no regression.**

Same model (`amd/DeepSeek-R1-0528-MXFP4-Preview`, 82 shards verified),
TP4 / 8k1k / CONC=128 / 384 prompts, 384/384 completed.
Model served from `/mnt/disk/huggingface/DeepSeek-R1-0528-MXFP4-Preview`.

Note: the model lives outside `/dev/shm`, so the container needs its parent
directory bind-mounted — otherwise HF treats the path as a repo id and fails
with `HFValidationError`.
