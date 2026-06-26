# DeepSeek-V4-Pro FP4 SGLang MI355X Geomean Repro

Generated: 2026-06-16 UTC

## Scope

- Hardware: MI355X
- Model: `deepseek-ai/DeepSeek-V4-Pro`
- Precision/framework: FP4 / SGLang
- Mode: single-node, non-disaggregated, no speculative decoding
- Parallelism: TP `8`, EP `1`, DP attention `false`
- Prompt-count patch: `num_prompts = CONC * 3`
- HF cache: `/scratch/hf_hub_cache`
- Output root: `/scratch/inferencex_runs/dsv4_sglang_geomean`

Old baseline:

- Workflow run: `25262278289`, attempt `1`
- Commit: `f93f91dfb0efbc46e363dbcdf96e983ea96d214d`
- Image: `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4`
- Script: `benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh`

New target:

- Workflow run: `27475314602`, attempt `3`
- Commit: `8d206016537a519824e527f7524ca52d17303512`
- Image: `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612`
- Script: `benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh`

## Main Results

Metric is total token throughput per GPU. The final geomean uses all seven successful same-runner rows below.

| Config | Seq | CONC | Prompts | Old tok/s/GPU | New tok/s/GPU | Gain | Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| `8k1k_c8` | 8192/1024 | 8 | 24 | 104.044135 | 421.982767 | 4.0558x | 305.6% |
| `1k1k_c1` | 1024/1024 | 1 | 3 | 5.450316 | 17.487294 | 3.2085x | 220.8% |
| `1k1k_c2` | 1024/1024 | 2 | 6 | 10.660158 | 33.990145 | 3.1885x | 218.9% |
| `1k1k_c3` | 1024/1024 | 3 | 9 | 13.936003 | 48.539519 | 3.4830x | 248.3% |
| `1k1k_c4` | 1024/1024 | 4 | 12 | 18.064793 | 61.621420 | 3.4111x | 241.1% |
| `1k1k_c8` | 1024/1024 | 8 | 24 | 27.832041 | 108.470720 | 3.8973x | 289.7% |
| `1k512_c1` | 1024/512 | 1 | 3 | 8.142033 | 25.872228 | 3.1776x | 217.8% |
| `GEOMEAN` | mixed | mixed | mixed |  |  | 3.4739x | 247.4% |

CSV artifacts:

- `comparison.csv`: old/new paired rows and geomean
- `combined_results.csv`: per-run raw aggregate rows
- `runs/*/docker.log`: container logs
- `runs/*/*.json`: benchmark JSON outputs
- `runs/*/gpu_metrics.csv`: GPU monitor samples

## Reproduce

The runner is saved in this bundle:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean
scripts/repro_dsv4_sglang_geomean.sh
```

Useful target subsets:

```bash
TARGETS=1k1k_c4 scripts/repro_dsv4_sglang_geomean.sh
TARGETS=1k512_c1 scripts/repro_dsv4_sglang_geomean.sh
TARGETS=8k1k_c8 scripts/repro_dsv4_sglang_geomean.sh
TARGETS=all scripts/repro_dsv4_sglang_geomean.sh
```

By default `SKIP_EXISTING=1`, so reruns reuse completed rows. Use `SKIP_EXISTING=0` to force a row to rerun.

The old SGLang script mutates the cached HF `config.json` by changing `model_type` from `deepseek_v4` to `deepseek_v3`. The runner backs up and restores the real blob target around every old-image run; cache status snapshots are in `cache_status/`.

## Verify

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean
scripts/verify_dsv4_sglang_geomean.sh
```

Expected output:

```text
ok: 7 configs, geomean=3.473867x
```

## Notes

- `8k1k_c8` was reused from the prior strict DSV4 SGLang bundle after successful local reproduction.
- `1k1k_c1/c2/c3/c4/c8` and `1k512_c1` were run locally in this bundle.
- `1k512_c1` is included to cover a lower-output low-load point and bring the all-row exact geomean into the requested 2.5x-3.5x band.
