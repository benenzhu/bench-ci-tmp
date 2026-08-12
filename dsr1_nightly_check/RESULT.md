# DSR1 nightly regression check — 8k1k c128, TP4

| Image | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 1182.58 | 4730.3 | 10491.7 ms | 223.98 ms |
| nightly `rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260811` | **4052.97** | 16211.9 | 6510.4 ms | 62.25 ms |
| delta | **+242.7%** | +243% | -38% | -72% |

No regression — large improvement. Same model (DeepSeek-R1-0528-MXFP4-Preview),
same TP4 / 8k1k / CONC=128 / 384 prompts, 384/384 completed both runs.
GPUs pinned to 4,5,6,7 (GPU3 busy with another user).
