# Kimi-K2.5 nightly regression check — 8k1k c128, TP8

| Image | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline `vllm/vllm-openai-rocm:v0.22.0` | 2239.96 | 17919.7 | 5141.8 ms | 56.86 ms |
| nightly `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | 2236.69 | 17893.5 | 6897.8 ms | 55.22 ms |
| delta | **-0.1%** | -0.1% | +34% | -2.9% |

**No regression** (within noise). Same model (amd/Kimi-K2.5-MXFP4), same
TP8 / 8k1k / CONC=128 / 384 prompts, 384/384 completed on both runs.
TTFT is higher but decode (TPOT) is marginally better; steady-state
throughput is unchanged.
