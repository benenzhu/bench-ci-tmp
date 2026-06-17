# Machine Release Handoff

Context as of 2026-06-17 UTC: assume this machine will be released and only this git repository/branch remains available. Do not rely on `/root`, `/scratch`, `/dev/shm`, Docker local state, or Hugging Face cache surviving unless an external snapshot is made outside this repo.

## Surviving State

The intended surviving state is the remote git branch:

```bash
git clone -b zz3 git@github.com:benenzhu/bench-ci-tmp.git /scratch/inferencex_runs
cd /scratch/inferencex_runs
```

This repo contains the benchmark results, logs, scripts, workflow metadata, comparison tables, verification scripts, and small inventories needed to restart work. It does not contain model weights or Docker image layers.

Primary entry points:

- `README.md`: top-level result summary and verification commands.
- `dsv4_sglang_geomean_prompts3x/`: final DSV4 geomean bundle. This is the cleanest current result for the requested 2.5x-3.5x geomean target.
- `dsv4_sglang_8k1k_conc8_prompts3x/`: strict DSV4 8k/1k CONC=8 bundle plus broader candidate docs.
- `kimi_int4_8k1k_conc16_prompts3x/`: Kimi INT4 local repro attempt that did not reproduce the dashboard 4x claim.
- `kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x/`: Kimi FP4 old-image flag ablation.
- `mi355_adsuite_prompts3x/`: original broader MI355X suite output.
- `backup_inventory/`: small inventories for the non-git state that must be recreated.

## Current Best Result To Use

Use the DSV4 SGLang geomean result for the "more configs, geomean 2.5x-3.5x" story:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean_prompts3x
scripts/verify_dsv4_sglang_geomean.sh
```

Expected:

```text
ok: 7 configs, geomean=3.473867x
```

Summary:

| Scope | Configs | Old image | New image | Geomean |
|---|---:|---|---|---:|
| DeepSeek-V4-Pro FP4 SGLang, MI355X, TP8, non-disagg, `num_prompts=CONC*3` | 7 | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | `3.473867x` / `247.4%` improvement |

Exact rows are in `dsv4_sglang_geomean_prompts3x/comparison.csv`.

## What Is Not In Git

Do not assume these survive the machine release:

- Hugging Face model cache: `/scratch/hf_hub_cache`, about `2.6T`.
- Hardlink model backup: `/scratch/model_backups`. It was useful on this machine but is on the same `/scratch` filesystem, so it does not help if only git survives.
- Docker image layers under `/var/lib/docker`, hundreds of GB.
- `/root/InferenceX` and temporary worktrees under `/scratch/inferencex_worktrees`.
- `/dev/shm` model copies or temporary cache.
- HF token. It was intentionally not written into repo files.

The repo stores inventories for these under `backup_inventory/`, not the actual large data.

## Model Cache To Recreate

Inventory files:

- `backup_inventory/model_cache/cache_summary.tsv`
- `backup_inventory/model_cache/blob_inventory.tsv`
- `backup_inventory/model_cache/snapshot_inventory.tsv`

Models that were present locally:

| Model | Snapshot | Approx size |
|---|---|---:|
| `deepseek-ai/DeepSeek-V4-Pro` | `5607980f3a4b8ea0371b9f11e1848ac41f14979e` | `806G` |
| `amd/Kimi-K2.5-MXFP4` | `419004c8716cf22c929aa15d39b85e09a8a2091a` | `521G` |
| `moonshotai/Kimi-K2.5` | `4d01dfe0332d63057c186e0b262165819efb6611` | `555G` |
| `zai-org/GLM-5-FP8` | `4f96cc5eec29dcee5d6ded54f7ffe889438f9516` | `705G` |

On a new machine, set a fresh token manually and download into a persistent cache:

```bash
export HF_HOME=/scratch/hf_home
export HF_HUB_CACHE=/scratch/hf_hub_cache
export MODEL_DIR=/scratch/hf_hub_cache
export HF_TOKEN=<set manually; do not commit>

python3 - <<'PY'
from huggingface_hub import snapshot_download

models = [
    ("deepseek-ai/DeepSeek-V4-Pro", "5607980f3a4b8ea0371b9f11e1848ac41f14979e"),
    ("amd/Kimi-K2.5-MXFP4", "419004c8716cf22c929aa15d39b85e09a8a2091a"),
    ("moonshotai/Kimi-K2.5", "4d01dfe0332d63057c186e0b262165819efb6611"),
    ("zai-org/GLM-5-FP8", "4f96cc5eec29dcee5d6ded54f7ffe889438f9516"),
]

for repo_id, revision in models:
    snapshot_download(repo_id=repo_id, revision=revision)
PY
```

For a single DSV4-only rerun, only `deepseek-ai/DeepSeek-V4-Pro` is required.

## Docker Images To Recreate

Inventory files:

- `backup_inventory/docker_images.tsv`
- `backup_inventory/docker_image_inspect/*.json`

Pull the tags needed for the target run. For DSV4 geomean:

```bash
docker pull rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4
docker pull lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612
```

Other tags captured in the inventory:

```text
lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413
lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260528
vllm/vllm-openai-rocm:v0.15.1
vllm/vllm-openai-rocm:v0.16.0
vllm/vllm-openai-rocm:v0.18.0
vllm/vllm-openai-rocm:v0.21.0
vllm/vllm-openai-rocm:v0.22.0
rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218
rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219
```

The local image IDs and inspect JSON are recorded for comparison, but the tags are the practical restore mechanism.

## Recreate Source Repo / Worktrees

Most repro scripts assume the InferenceX source repo exists at `/root/InferenceX`.

```bash
git clone git@github.com:SemiAnalysisAI/InferenceX.git /root/InferenceX
```

The scripts create detached worktrees at the exact old/new workflow commits and patch them locally for `num_prompts=CONC*3`. The patched scripts, local diffs, exact env files, raw JSON, aggregate JSON, server logs, docker logs, and GPU metrics are already preserved in each run directory.

## Verification After Clone

These checks do not require model weights or Docker images because they validate saved artifacts:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean_prompts3x
scripts/verify_dsv4_sglang_geomean.sh

cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
scripts/verify_bundle.sh

cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
scripts/verify_kimi_int4_8k1k_conc16.sh

cd /scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x
scripts/verify_kimi_fp4_old_flags_ablation.sh
```

Before pushing future changes, run a token scan:

```bash
rg -n 'hf_[A-Za-z0-9]{20,}' /scratch/inferencex_runs || true
```

Expected: no output.

## Rerun Commands

DSV4 geomean:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean_prompts3x
TARGETS=all scripts/repro_dsv4_sglang_geomean.sh
```

DSV4 strict 8k/1k:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
TARGETS=all scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh
```

Kimi INT4 attempted middle-state:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
TARGETS=all scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
```

Kimi FP4 old flags ablation:

```bash
cd /scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x
TARGETS=all scripts/repro_kimi_fp4_old_flags_ablation.sh
```

Legacy Qwen3.5 FP8 scratch script is archived at:

```text
backup_inventory/root_scripts/repro_qwen35_fp8_1k1k_tp8.sh
```

It was not part of the final DSV4/Kimi/GLM handoff results.

## Important Caveats

- The old DSV4 SGLang script mutates cached HF `config.json` from `deepseek_v4` to `deepseek_v3`. The DSV4 runners back up and restore that file/blob. Saved cache status confirms final `model_type=deepseek_v4`.
- The Kimi INT4 `8192/1024, CONC=16` local attempt completed but produced only `1.3200x`, not the recorded dashboard `4.0143x`; do not use it as a reproduced 4x claim.
- The Kimi FP4 full-new-flags-on-old-image run is blocked on old `v0.16.0` at TP8 because AITER MLA rejects 8 local heads.
- If someone needs actual model weights to survive without redownload, this repo is insufficient. Use external object storage, a volume snapshot, or an artifact store capable of multi-TB files. Ordinary git is intentionally only carrying reproducibility metadata and benchmark artifacts.
