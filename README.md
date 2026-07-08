# Summary

MI355X inference speedups from old→new image/runtime upgrades, reproduced locally. Each number is the geometric mean of per-config old→new `tok/s/GPU` gains across a concurrency sweep (per-GPU normalized, so TP1/TP4/TP8 are comparable).

| Model | Prec / Framework | ROCm (old → new) | Speedup (geomean) |
| --- | --- | --- | ---: |
| GLM-5 | FP8 / SGLang→ATOM | 7.0.0 → 7.2.2 | **3.52x** |
| Kimi-K2.5 | FP4 / vLLM | 7.0.0 → 7.2.2 | **3.71x** |
| DeepSeek-R1-0528 | FP4 / SGLang | 7.0.0 → 7.2.0 | **2.55x** |

## Key Optimizations

| # | Optimization | What it does |
|---|---|---|
| 1 | AITER optimized kernels | Fused & optimized GPU kernels for inference (MHC, fused hash-topk, fused compress) |
| 2 | MoE/Attention backend upgrades | FlyDSL MoE + Unified KV attention backends optimizations |
| 3 | Parallelism & scheduling tuning | Expert parallelism (EP), DP-attention, and two-batch overlap (TBO) |

## How to Reproduce

Full per-bundle instructions, exact per-config numbers, image/model revisions, and verify scripts are in **[DETAIL.md](./DETAIL.md)**.

- **GLM-5 FP8** (rocm700 → ATOM, 3.52x): `glm5_fp8_rocm700_geomean/scripts/repro_glm5_fp8_rocm700_geomean.sh`
- **Kimi-K2.5 FP4** (3.71x): `kimi_fp4_geomean/scripts/repro_kimi_fp4_geomean.sh`
- **DeepSeek-R1-0528 FP4** (2.55x): `dsr1_fp4_tp4_geomean/scripts/repro_dsr1_fp4_tp4_geomean.sh`

Each bundle has a `verify_*.sh` that re-checks the recorded geomean against its `comparison.csv`.
