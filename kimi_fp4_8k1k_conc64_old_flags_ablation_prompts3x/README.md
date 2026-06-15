# Kimi-K2.5 MXFP4 Old-Image Flag Ablation

Generated: 2026-06-15T10:11:33Z

Scope:

- Model: `amd/Kimi-K2.5-MXFP4`
- Old workflow: `22555187591/attempts/2`, job `65962398629`, commit `bc818474fdba`
- New reference workflow: `23614589869/attempts/1`, job `68778764973`
- Image under test: `vllm/vllm-openai-rocm:v0.16.0`
- Hardware/framework/precision: MI355X / vLLM / FP4
- Shape: ISL/OSL `8192/1024`, CONC `64`, TP `8`, EP `1`
- Prompt count patch: `num_prompts = CONC * 3 = 192`
- Dashboard reference for this same row: `348.5322178041199 -> 1647.2603593393117 tok/s/GPU`, gain `4.7263x`

Variants:

| label | note |
|---|---|
| `old_baseline` | old image with original 03-01 env/server flags |
| `old_all_new_flags` | old image plus 03-26 env/server flags: HSA conditional, AITER, INT4 quick reduce, gpu_memory=0.90, block_size=1, no prefix cache, no disable-log-requests |
| `old_server_flags_only` | old image plus server flags only: gpu_memory=0.90, block_size=1, no prefix cache, no disable-log-requests |
| `old_server_flags_no_block1` | old image plus compatible server flags only: gpu_memory=0.90, no prefix cache, no disable-log-requests; block_size remains 64 |
| `old_env_flags_only` | old image plus env flags only: HSA conditional, AITER, INT4 quick reduce; old server flags retained |
| `old_aiter_quick_only` | old image plus VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 only |
| `old_aiter_only` | old image plus VLLM_ROCM_USE_AITER=1 only |
| `old_block1_only` | old image plus --block-size=1 only |
| `old_no_prefix_only` | old image plus --no-enable-prefix-caching only |
| `old_mem090_only` | old image plus --gpu-memory-utilization 0.90 only |

## Summary

| Case | Outcome |
|---|---|
| Full 03-26 env/server flags on the old image | Failed before benchmark. The old `v0.16.0` image hits `Aiter MLA only supports 16 or 128 number of heads. Provided 8 number of heads.` |
| `VLLM_ROCM_USE_AITER=1` only | Failed with the same AITER MLA assertion. |
| `--block-size=1` only | Failed with the same AITER MLA assertion, even without explicitly exporting `VLLM_ROCM_USE_AITER=1`. |
| Compatible server subset, keeping `block_size=64` | Completed 192 requests at `290.762867 tok/s/GPU` total throughput, `32.307676 output tok/s/GPU`, duration `683.309s`. |

Interpretation: the full new 03-26 Kimi FP4 configuration is not runnable on the old 03-01 image at TP8 for this model shape. The two blocking knobs are AITER MLA and `--block-size=1`; both route the old image into an AITER MLA path that rejects TP8 because the local head count is 8. The only measured old-image compatibility run keeps `block_size=64` and disables AITER while retaining `gpu_memory=0.90`, `--no-enable-prefix-caching`, and removal of `--disable-log-requests`.

## Results

| label | status | flags | tok/s/GPU | output tok/s/GPU | duration s | completed |
|---|---|---|---:|---:|---:|---:|
| `old_baseline` | not_run | hsa=, aiter=, quick=none, mem=, block=, no_prefix=, disable_log= |  |  |  |  |
| `old_all_new_flags` | docker_failed:1 | hsa=1, aiter=1, quick=INT4, mem=0.90, block=1, no_prefix=1, disable_log=0 |  |  |  |  |
| `old_server_flags_only` | docker_failed:1 | hsa=0, aiter=0, quick=none, mem=0.90, block=1, no_prefix=1, disable_log=0 |  |  |  |  |
| `old_server_flags_no_block1` | success | hsa=0, aiter=0, quick=none, mem=0.90, block=64, no_prefix=1, disable_log=0 | 290.762867 | 32.307676 | 683.308970 | 192 |
| `old_env_flags_only` | not_run | hsa=, aiter=, quick=none, mem=, block=, no_prefix=, disable_log= |  |  |  |  |
| `old_aiter_quick_only` | not_run | hsa=, aiter=, quick=none, mem=, block=, no_prefix=, disable_log= |  |  |  |  |
| `old_aiter_only` | docker_failed:1 | hsa=0, aiter=1, quick=none, mem=0.95, block=64, no_prefix=0, disable_log=1 |  |  |  |  |
| `old_block1_only` | docker_failed:1 | hsa=0, aiter=0, quick=none, mem=0.95, block=1, no_prefix=0, disable_log=1 |  |  |  |  |
| `old_no_prefix_only` | not_run | hsa=, aiter=, quick=none, mem=, block=, no_prefix=, disable_log= |  |  |  |  |
| `old_mem090_only` | not_run | hsa=, aiter=, quick=none, mem=, block=, no_prefix=, disable_log= |  |  |  |  |

## Reproduce

Actual cache paths used for this run:

```bash
HF_HUB_CACHE=/scratch/hf_hub_cache
HF_HOME=/scratch/hf_home
```

Rerun the measured compatible subset:

```bash
cd /root
TARGETS=old_server_flags_no_block1 SKIP_EXISTING=0 \
  /scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x/scripts/repro_kimi_fp4_old_flags_ablation.sh
```

Rerun the failing bisection points without stopping at the first failure:

```bash
TARGETS=old_all_new_flags,old_aiter_only,old_block1_only \
  SKIP_EXISTING=0 CONTINUE_ON_ERROR=1 \
  /scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x/scripts/repro_kimi_fp4_old_flags_ablation.sh
```

To force a different model cache, set `MODEL_DIR`, for example `MODEL_DIR=/dev/shm` if there is enough RAM-backed space.

## Validate

```bash
cd /scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x
cat combined_results.csv
rg -n "Aiter MLA only supports|Provided 8 number of heads|Using Triton MLA backend" runs/*/*.log
python3 - <<'PY'
import json
from pathlib import Path
p = next(Path("runs/old_server_flags_no_block1").glob("agg_*.json"))
data = json.loads(p.read_text())
print(data["tput_per_gpu"], data["output_tput_per_gpu"])
PY
```
