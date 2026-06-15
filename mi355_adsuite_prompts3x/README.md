# MI355X Ad Benchmark Repro

Generated: 2026-06-15T05:04:26Z

Scope:

- Hardware: MI355X
- Mode: single-node non-disaggregated
- Spec decoding: none
- Prompt count patch: `num_prompts = CONC * 3`
- HF cache: `/scratch/hf_hub_cache`
- HF home/xet cache: `/scratch/hf_home`
- Output root: `/scratch/inferencex_runs/mi355_adsuite_prompts3x`

Targets:

| label | model | precision | framework | image | commit | seq | conc | tp | workflow run | job |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| kimi_old | `amd/Kimi-K2.5-MXFP4` | fp4 | vllm | `vllm/vllm-openai-rocm:v0.16.0` | `bc818474fdba` | 8192/1024 | 4 | 8 | 22555187591 attempt 2 | 65962398410 |
| kimi_new | `amd/Kimi-K2.5-MXFP4` | fp4 | vllm | `vllm/vllm-openai-rocm:v0.22.0` | `a187fa20e778` | 8192/1024 | 4 | 8 | 26696253037 attempt 1 | 78682149908 |
| dsv4_old | `deepseek-ai/DeepSeek-V4-Pro` | fp4 | vllm | `vllm/vllm-openai-rocm:v0.21.0` | `ac04da00630f` | 1024/1024 | 4 | 8 | 26110663181 attempt 1 | 76787086168 |
| dsv4_new | `deepseek-ai/DeepSeek-V4-Pro` | fp4 | vllm | `vllm/vllm-openai-rocm:v0.22.0` | `4e52736885b6` | 1024/1024 | 4 | 8 | 26696268345 attempt 1 | 78681278252 |
| glm_old | `zai-org/GLM-5-FP8` | fp8 | sglang | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` | `f3d0fc5bd7b1` | 1024/1024 | 4 | 8 | 22792161490 attempt 5 | 66170603755 |
| glm_new | `zai-org/GLM-5-FP8` | fp8 | sglang | `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413` | `6a3ca2d29379` | 1024/1024 | 4 | 8 | 24574269364 attempt 1 | 71855382514 |

Image notes:

- `dsv4_old`: original workflow image `vllm/vllm-openai-rocm:nightly-b50646e5effd7cb5884cd96fdff4c53c18521198`; running image `vllm/vllm-openai-rocm:v0.21.0`. original workflow image tag is no longer available from Docker Hub; using v0.21.0 released 2026-05-15 as the closest public release before the 2026-05-19 workflow.

Runtime patch notes:

- `dsv4_old`: patch vLLM 0.21.0 MHC TileLang kernels inside the ephemeral container to disable PDL calls on ROCm; without this, server startup fails with 'PDL is not supported'.

## Results

Measured with `num_prompts = CONC * 3`, `CONC=4`, `TP=8`, non-disaggregated:

| Model | Precision | Framework | Seq | Old label/status | Old tok/s/GPU | New label/status | New tok/s/GPU | Measured gain | Dashboard expected |
|---|---|---|---|---|---:|---|---:|---:|---:|
| Kimi-K2.5 | fp4 | vLLM | 8192/1024 | `kimi_old` success | 24.52 | `kimi_new` success | 359.12 | 14.65x (+1364.8%) | 28.7 -> 374.2 |
| DeepSeek-V4-Pro | fp4 | vLLM | 1024/1024 | `dsv4_old` failed | n/a | `dsv4_new` success | 21.33 | n/a | 3.3 -> 23.3 |
| GLM-5 | fp8 | SGLang | 1024/1024 | `glm_old` success | 16.17 | `glm_new` success | 41.52 | 2.57x (+156.8%) | 17.5 -> 44.2 |

`dsv4_old` is not an exact local reproduction. The original workflow image `vllm/vllm-openai-rocm:nightly-b50646e5effd7cb5884cd96fdff4c53c18521198` was unavailable from Docker Hub, so this suite used `vllm/vllm-openai-rocm:v0.21.0` as the closest public substitute. After the local PDL-disable runtime patch, it still failed before benchmark with `Cannot find global function target.build.tilelang_hip`.

Summary files:

```text
/scratch/inferencex_runs/mi355_adsuite_prompts3x/combined_results.csv
/scratch/inferencex_runs/mi355_adsuite_prompts3x/comparison.csv
```

Per-run artifacts:

```text
/scratch/inferencex_runs/mi355_adsuite_prompts3x/runs/<label>/
```

Each run directory contains the available `docker.log`, `server.log`, `gpu_metrics.csv`, raw benchmark JSON, aggregate JSON, `env.repro`, `local_patch.diff`, and `status.txt`.

Reproduction entrypoints:

```text
/root/repro_mi355_adsuite.sh
/scratch/inferencex_runs/mi355_adsuite_prompts3x/repro_commands.sh
```

Cache / backup status:

```text
/scratch/hf_hub_cache/models--amd--Kimi-K2.5-MXFP4
/scratch/hf_hub_cache/models--deepseek-ai--DeepSeek-V4-Pro
/scratch/hf_hub_cache/models--zai-org--GLM-5-FP8
/scratch/inferencex_runs/mi355_adsuite_prompts3x/cache_status/
```
