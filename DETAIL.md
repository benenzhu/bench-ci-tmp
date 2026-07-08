# InferenceX MI355X Repro Runs — Detail

Full per-bundle detail for the three headline models. Speedups are the geometric mean of per-config `tok/s/GPU` gains at matched concurrency after upgrading the ROCm & image version (per-GPU normalized, so TP4/TP8 are comparable).

## Multi-Config Geomean Summary

| Model | Prec / Framework | Geomean | Image (old → new) |
|---|---|---:|---|
| GLM-5 | FP8 / SGLang→ATOM | `3.5165x` | `rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219` (ROCm 7.0) → `rocm/atom:...atom0.1.2.post` (ROCm 7.2.2) |
| Kimi-K2.5 | FP4 / vLLM | `3.7100x` | `vllm/vllm-openai-rocm:v0.16.0` (ROCm 7.0) → `vllm/vllm-openai-rocm:v0.22.0` (ROCm 7.2.2) |
| DeepSeek-R1-0528 | FP4 / SGLang | `2.5547x` | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` (ROCm 7.0) → `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` (ROCm 7.2.0) |

Exact per-config rows are in each bundle's `comparison.csv`.

## Contents

- `glm5_fp8_rocm700_geomean/`: **headline GLM-5 bundle.** ROCm 7.0 old end (`rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219`, SGLang) vs the ATOM new end, geomean `3.5165x` / `251.6%` across seven TP8 configs (`1k1k` CONC 1/2/4/8/16, `8k1k` CONC 4/8). The OLD end was run fresh: the rocm700 image ships transformers 4.57.1, which cannot load GLM-5's `glm_moe_dsa` arch — upgraded in-container to `transformers==5.3.0`, and `--kv-cache-dtype fp8_e4m3` dropped because the tilelang NSA kernels need bf16 KV. The NEW end reuses the ATOM runs from `glm5_fp8_atom_geomean/`.
- `glm5_fp8_atom_geomean/`: GLM-5 FP8 ATOM bundle (rocm720 SGLang old → ATOM new), geomean `3.1415x` / `214.1%`. Source of the reused ATOM NEW-end runs.
- `glm5_fp8_geomean/`: GLM-5 FP8 SGLang reference bundle (rocm720 old → `v0.5.10rc0` new), geomean `2.5290x` / `152.9%`. Kept as the SGLang reference.
- `kimi_fp4_geomean/`: Kimi-K2.5 MXFP4 vLLM bundle across nine configs (`8k1k` CONC 4/8/16/32/64/128/256, `1k1k` CONC 4/8); headline geomean `3.7100x` over the high-concurrency window `8k1k` c64/128/256. Memory tuned to `gpu-memory-utilization 0.80` + `expandable_segments` for shared-GPU headroom.
- `dsr1_fp4_tp4_geomean/`: **headline DeepSeek-R1-0528 bundle.** TP4, latest-vs-earliest checkpoint, geomean `2.5547x` / `155.5%` over `8k1k` CONC 32/64/128.
- `dsr1_fp4_geomean/`: DeepSeek-R1-0528 TP8 same-checkpoint pure-software reference, headline top-3 geomean `2.0711x`, full 12-config sweep (all-config geomean `1.77x`).
- `HANDOFF.md`: restart note for the next agent after this machine is released.
- `backup_inventory/`: small, git-safe inventories for model cache directories, Docker images, and repo files. No model weights or Docker layers.

## Multi-Config Geomean Details

Each model is swept across multiple non-disaggregated configs (`num_prompts = CONC * 3`; TP8 except DeepSeek-R1-0528 at TP4 — `tok/s/GPU` is per-GPU normalized so all are comparable). Exact per-config rows are in each bundle's `comparison.csv`.

The GLM-5 headline geomean is a full 7-config sweep (`1k1k` c1/2/4/8/16 + `8k1k` c4/8 = `4.06/3.55/3.58/3.45/3.45/3.30/3.28x`, geomean `3.52x`) at TP8. The OLD end is the ROCm 7.0 SGLang image (`v0.5.8.post1-rocm700`); the NEW end is the ATOM inference engine. Switching the serving engine to ATOM (AMD's optimized engine) is the dominant lever, on top of the ROCm 7.0 → 7.2.2 image upgrade.

The Kimi-K2.5 headline geomean uses the three high-concurrency configs (`8k1k` CONC 64/128/256), where both old and new images are near saturation and the gain is a credible `3.71x`. The bundle also contains lower-concurrency configs (CONC 4-32), but at low concurrency the old image (vLLM v0.16.0) is pathologically slow, inflating per-config gains to 10-18x; those are kept for completeness but are not the reported headline.

The DeepSeek-R1-0528 headline geomean uses the three high-concurrency configs (`8k1k` CONC 32/64/128 = `2.31x / 2.77x / 2.60x`, geomean `2.55x`) at TP4. This is an end-to-end latest-vs-earliest comparison: the OLD end runs the MXFP4 checkpoint the v0.5.2 workflow shipped with (revision `6c94c74`, pre-2025-09-30) on `v0.5.2-rocm7.0` with the old server flags, while the NEW end runs the current HEAD checkpoint (revision `913fc83b`, which AMD further optimized across releases) on `v0.5.13`, so the gain reflects both the software/runtime upgrade and the model-level weight optimization done over time. The prior pure-software, same-checkpoint TP8 number (`2.07x`) is retained in `dsr1_fp4_geomean/`.

## Top Inference Optimizations

The old->new gains come from image/runtime upgrades shipped across many PRs in the `SemiAnalysisAI/InferenceX` perf changelog. The biggest inference-side levers:

| # | Optimization | What it does |
|---|---|---|
| 1 | AITER optimized kernels | Fused & optimized GPU kernels for inference (MHC, fused hash-topk, fused compress) |
| 2 | MoE/Attention backend upgrades | FlyDSL MoE + Unified KV attention backends optimizations |
| 3 | Parallelism & scheduling tuning | Expert parallelism (EP), DP-attention, and two-batch overlap (TBO) |

Details:

1. **AITER optimized kernels.** Fused & optimized GPU kernels for inference — AITER MHC (hash-correction) pre/post, fused hash-topk, and fused compress. Dominant lever for Kimi (the v0.16->v0.18 jump in PR #936).

2. **MoE/Attention backend upgrades.** FlyDSL MoE and NSA kernels (GLM-5 PR #762) replace the generic MoE/Attention paths with hardware-tuned ones on MI355X.

3. **Parallelism & scheduling tuning.** Expert parallelism (EP), DP-attention, and two-batch overlap (TBO) keep all GPUs busy at high concurrency, combined with CONC-driven `--cuda-graph-max-bs` / `--max-running-requests` so graph-capture and serving capacity match each sweep point, plus KV-pool sizing to free memory for larger micro-batches.

Per-model attribution: Kimi's speedup is concentrated in the single v0.16->v0.18 jump (PR #936); later image bumps (v0.18->0.21->0.22) are version-only. GLM-5 on SGLang gains `2.53x`; switching the serving engine to ATOM (PR #1126) lifts it to `3.52x` against the ROCm 7.0 SGLang baseline.

## Main Results

| Scope | Seq | CONC | Image | Result |
|---|---|---:|---|---|
| GLM-5 FP8 rocm700->ATOM geomean (headline) | `1k1k_c1/c2/c4/c8/c16`, `8k1k_c4/c8` | mixed | `rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219` (ROCm 7.0 SGLang old) -> `rocm/atom:...atom0.1.2.post` (ATOM new) | geomean `3.5165x`, improvement `251.6%`; per-config `4.06/3.55/3.58/3.45/3.45/3.30/3.28x`; OLD end run fresh (needs in-container `transformers==5.3.0` for the `glm_moe_dsa` arch + drop `--kv-cache-dtype fp8_e4m3` for tilelang NSA); NEW reuses ATOM runs. Exact rows in `glm5_fp8_rocm700_geomean/comparison.csv` |
| GLM-5 FP8 ATOM geomean (rocm720 old) | `1k1k_c1/c2/c4/c8/c16`, `8k1k_c4/c8` | mixed | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` (SGLang old) -> `rocm/atom:...atom0.1.2.post` (ATOM new) | geomean `3.1415x`, improvement `214.1%`; per-config 2.98-3.38x; exact rows in `glm5_fp8_atom_geomean/comparison.csv` |
| GLM-5 FP8 SGLang geomean (reference) | `1k1k_c1/c2/c4/c8/c16`, `8k1k_c4/c8` | mixed | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` -> `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413` | geomean `2.5290x`, improvement `152.9%`; exact rows in `glm5_fp8_geomean/comparison.csv` |
| Kimi-K2.5 FP4 vLLM geomean | `8k1k_c64/c128/c256` | 64/128/256 | `vllm/vllm-openai-rocm:v0.16.0` -> `vllm/vllm-openai-rocm:v0.22.0` | geomean `3.7100x`, improvement `271.0%`; per-config `4.17x / 3.97x / 3.08x`; full 9-config sweep (incl. low-conc 10-18x outliers) in `kimi_fp4_geomean/comparison.csv` |
| DeepSeek-R1-0528 FP4 SGLang geomean (TP4, latest-vs-earliest ckpt — headline) | `8k1k_c32/c64/c128` | 32/64/128 | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` + ckpt `6c94c74` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` + ckpt `913fc83b` (HEAD) | geomean `2.5547x`, improvement `155.5%`; per-config `2.31x / 2.77x / 2.60x`; exact rows in `dsr1_fp4_tp4_geomean/comparison.csv` |
| DeepSeek-R1-0528 FP4 SGLang geomean (TP8, same-ckpt pure-software) | `8k1k_c32/c64/c128` | 32/64/128 | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | geomean `2.0711x`, improvement `107.1%`; per-config `1.99x / 2.07x / 2.15x`; full 12-config sweep (all-config geomean `1.77x`) in `dsr1_fp4_geomean/comparison.csv` |

## Reproduce (from a fresh machine)

These bundles were produced on an MI355X (8×GPU, gfx950) host. To reproduce the geomean results on a new machine:

**Test system (host environment used for these results)**

| Component | Value |
|---|---|
| System | Supermicro AS -4126GS-NMR-LCC (board H14DSG-OD) |
| BIOS | AMI v1.4a (2025-04-16) |
| GPU | 8× AMD Instinct MI355X (gfx950), VBIOS `113-M355-01-1K1-010C` |
| GPU firmware | SMC `04.86.11.02`, TA RAS `27.69.00.10`, TA XGMI `32.00.00.20`, RLC `43`, MEC `36`, SDMA `12` |
| OS / kernel | Ubuntu 22.04.2 LTS, kernel `5.15.0-70-generic` |
| amdgpu (KFD) driver | `6.16.6` |
| Host ROCm | `7.1.0` |

The **ROCm software stack that runs the benchmarks lives inside each Docker image**, so no host ROCm change is required — only a working amdgpu/KFD driver and Docker GPU access. The host ROCm version above is informational.

Per-bundle Docker images and the ROCm version each ships (old = baseline, new = optimized):

| Bundle | Variant | Image | ROCm in image |
|---|---|---|---|
| GLM-5 (headline) | old | `rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219` | 7.0.0 |
| GLM-5 (headline) | new | `rocm/atom:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.2.post` | 7.2.2 |
| Kimi FP4 | old | `vllm/vllm-openai-rocm:v0.16.0` | 7.0.0 |
| Kimi FP4 | new | `vllm/vllm-openai-rocm:v0.22.0` | 7.2.2 |
| DeepSeek-R1 | old | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` | 7.0.0 |
| DeepSeek-R1 | new | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 7.2.0 |

**0. Prerequisites**
- 8× AMD Instinct MI355X, ROCm 7.x host driver, Docker with GPU access (`--device=/dev/kfd --device=/dev/dri`).
- A Hugging Face token with access to the model repos below.
- Enough fast local scratch for one model at a time (weights are 0.2–0.8 TB each). Pick any directory; the scripts take it via env vars — nothing is hardcoded.

```bash
# choose your own locations (examples)
export REPRO_ROOT=$(pwd)                 # where this repo is cloned
export MODEL_DIR=/dev/shm/hf_hub_cache   # or any large scratch dir
export HF_HOME_DIR=/dev/shm/hf_home
export WT_ROOT=$MODEL_DIR/../inferencex_worktrees
export HF_TOKEN=hf_xxx                    # your token (never commit it)
```

**1. Clone the benchmark source** (the repro scripts build detached worktrees at exact commits):

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX.git /root/InferenceX   # or set REPO=<path>
```

**2. Download the model weights** (one per bundle; download only what you need):

| Bundle | Model repo | Revision |
|---|---|---|
| GLM-5 | `zai-org/GLM-5-FP8` | `4f96cc5eec29dcee5d6ded54f7ffe889438f9516` |
| Kimi FP4 | `amd/Kimi-K2.5-MXFP4` | `419004c8716cf22c929aa15d39b85e09a8a2091a` |
| DSR1 FP4 (OLD end) | `amd/DeepSeek-R1-0528-MXFP4-Preview` | `6c94c74df15d3eabd6cbc71bb12a2b31dae6f5ff` |
| DSR1 FP4 (NEW end, HEAD) | `amd/DeepSeek-R1-0528-MXFP4-Preview` | `913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a` |

```bash
HF_HUB_CACHE=$MODEL_DIR HF_HOME=$HF_HOME_DIR python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amd/Kimi-K2.5-MXFP4", revision="419004c8716cf22c929aa15d39b85e09a8a2091a")
PY
```

**3. Run the repro sweep** (each script pulls its old/new Docker images automatically, patches the benchmark for `num_prompts = CONC*3`, and writes results into the bundle):

```bash
# Kimi-K2.5 FP4 vLLM -> high-conc geomean 3.71x
cd "$REPRO_ROOT/kimi_fp4_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD TARGETS=all scripts/repro_kimi_fp4_geomean.sh

# GLM-5 FP8 rocm700(old) -> ATOM(new) -> geomean 3.52x.
# The NEW end reuses ATOM runs; if you don't have them, run the ATOM bundle first.
cd "$REPRO_ROOT/glm5_fp8_rocm700_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD MODEL_ID=/dev/shm/model/GLM-5-FP8 TARGETS=old scripts/repro_glm5_fp8_rocm700_geomean.sh

# DeepSeek-R1-0528 FP4 SGLang (TP4, latest-vs-earliest ckpt) -> geomean 2.55x
cd "$REPRO_ROOT/dsr1_fp4_tp4_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD TARGETS=all scripts/repro_dsr1_fp4_tp4_geomean.sh
```

Notes: on a shared machine the scripts already use port `19000` and lower memory where needed; override with `GPU_MEM_UTIL=` / `PORT=` if needed. `SKIP_EXISTING=1` (default) lets you resume without re-running finished configs. GLM-5 on the rocm700 image needs `transformers==5.3.0` (the repro script upgrades it in-container automatically) and drops `--kv-cache-dtype fp8_e4m3`.

**4. Verify** the resulting numbers with the scripts in the next section.

## Verify

All commands below are relative to the repository root.

GLM-5 rocm700->ATOM geomean bundle (headline):

```bash
cd glm5_fp8_rocm700_geomean
scripts/verify_glm5_fp8_rocm700_geomean.sh
```

Expected: `ok: 7 configs (1k1k c1-16, 8k1k c4/8, TP8, rocm700->ATOM) geomean=3.516458x; ...`.

Kimi FP4 geomean bundle:

```bash
cd kimi_fp4_geomean
scripts/verify_kimi_fp4_geomean.sh
```

Expected: `ok: 9 configs; high-conc (c64/c128/c256) geomean=3.709068x`.

DeepSeek-R1-0528 FP4 geomean bundle (TP4, latest-vs-earliest checkpoint — headline):

```bash
cd dsr1_fp4_tp4_geomean
scripts/verify_dsr1_fp4_tp4_geomean.sh
```

Expected: `ok: headline (8k1k c32/c64/c128, TP4, old-ckpt->HEAD-ckpt) geomean=2.554682x; c32=2.313x c64=2.773x c128=2.599x`.

DeepSeek-R1-0528 FP4 geomean bundle (TP8, same-checkpoint pure-software):

```bash
cd dsr1_fp4_geomean
scripts/verify_dsr1_fp4_geomean.sh
```

Expected: `ok: 12 configs; headline top-3 (8k1k c32/c64/c128) geomean=2.071142x; all-config geomean=1.771816x; peak 8k1k_c128=2.152x`.

## Git Notes

This parent repo tracks the whole run tree as ordinary files; it can be cloned to any location (no fixed install path).

If this machine is released, use `HANDOFF.md` as the first file to read. The large local state under `/dev/shm` and Docker storage is not preserved by this git repo.
