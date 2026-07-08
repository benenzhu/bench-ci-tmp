# Summary

MI355X inference speedups from old→new image/runtime upgrades, reproduced locally. Each number is the geometric mean of per-config old→new `tok/s/GPU` gains across a concurrency sweep (per-GPU normalized, so TP1/TP4/TP8 are comparable).

| Model | Prec / Framework | New image | Old image | Speedup (geomean) | Breakdowns |
| --- | --- | --- | --- | ---: | --- |
| GLM-5 | FP8 / SGLang→ATOM | `rocm/atom:...atom0.1.2.post` (ROCm 7.2.2) | `rocm/sgl-dev:v0.5.8.post1-rocm700` (ROCm 7.0) | **3.52x** | 7 configs `3.28–4.06x`; ATOM engine vs SGLang baseline |
| ~~DeepSeek-V4-Pro~~ | ~~FP4 / SGLang~~ | ~~`lmsysorg/sglang-rocm:v0.5.13-rocm720`~~ | ~~`rocm/sgl-dev:rocm720-mi35x-...-DSv4`~~ | ~~**3.47x**~~ | ~~7 configs~~ |
| Kimi-K2.5 | FP4 / vLLM | `vllm/vllm-openai-rocm:v0.22.0` | `vllm/vllm-openai-rocm:v0.16.0` | **3.71x** | high-conc `8k1k` c64/128/256 |
| DeepSeek-R1-0528 | FP4 / SGLang | `lmsysorg/sglang-rocm:v0.5.13-rocm720` + HEAD ckpt | `...v0.5.2-rocm7.0` + `6c94c74` ckpt | **2.55x** | `8k1k` c32/64/128; latest-vs-earliest ckpt |
| gpt-oss-120b | FP4 / vLLM | `vllm/vllm-openai-rocm:v0.22.0` | `rocm/7.0:...vllm_0.10.1` | **1.15x** | 5 configs; generic MoE, no DeepSeek kernels |

## Key Optimizations

- **AITER optimized kernels**: fused & tuned GPU kernels — AITER MHC (hash-correction) pre/post, fused hash-topk, fused compress. Dominant lever for Kimi and DeepSeek-V4-Pro.
- **MoE / Attention backend upgrades**: FlyDSL MoE, unified-KV / NSA attention backends replace generic paths with MI355X-tuned ones.
- **ATOM inference engine**: AMD's optimized engine (vs SGLang) is the headline lever for GLM-5.
- **Parallelism & scheduling tuning**: expert parallelism (EP), DP-attention, two-batch overlap (TBO), and CONC-driven graph/serving/KV-pool sizing to keep all GPUs busy at high concurrency.

## How to Reproduce

Full per-bundle instructions, exact per-config numbers, image/model revisions, and verify scripts are in **[DETAIL.md](./DETAIL.md)**.

- **GLM-5 FP8** (rocm700 → ATOM, 3.52x): `glm5_fp8_rocm700_geomean/scripts/repro_glm5_fp8_rocm700_geomean.sh`
- **Kimi-K2.5 FP4** (3.71x): `kimi_fp4_geomean/scripts/repro_kimi_fp4_geomean.sh`
- **DeepSeek-R1-0528 FP4** (2.55x): `dsr1_fp4_tp4_geomean/scripts/repro_dsr1_fp4_tp4_geomean.sh`
- **gpt-oss-120b FP4** (1.15x): `gptoss_120b_fp4_geomean/scripts/repro_gptoss_120b_fp4_geomean.sh`

Each bundle has a `verify_*.sh` that re-checks the recorded geomean against its `comparison.csv`.
