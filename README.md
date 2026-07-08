# InferenceX MI355X Repro Runs

## Multi-Config Geomean Summary

Geometric mean of per-config old->new `tok/s/GPU` gains on MI355X (per-GPU normalized, so TP1/TP4/TP8 are comparable):

| Model | Prec / Framework | Geomean | Image (old → new) |
|---|---|---:|---|
| DeepSeek-V4-Pro | FP4 / SGLang | `3.4739x` | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` → `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` |
| Kimi-K2.5 | FP4 / vLLM | `3.7100x` | `vllm/vllm-openai-rocm:v0.16.0` → `vllm/vllm-openai-rocm:v0.22.0` |
| GLM-5 | FP8 / SGLang→ATOM | `3.1415x` | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` → `rocm/atom:...atom0.1.2.post` |
| DeepSeek-R1-0528 | FP4 / SGLang | `2.5547x` | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` → `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` |
| gpt-oss-120b | FP4 / vLLM | `1.1507x` | `rocm/7.0:...vllm_0.10.1_instinct_20250927_rc1` → `vllm/vllm-openai-rocm:v0.22.0` |

This repository contains the local MI355X benchmark reproduction artifacts produced on 2026-06-15 UTC.

## Contents

- `dsv4_sglang_8k1k_conc8_prompts3x/`: primary handoff bundle. It includes the strict same-config DeepSeek-V4-Pro FP4 SGLang reproduction, the broader MI355X Kimi/GLM suite copied under `suites/`, candidate tables, logs, raw JSON, aggregate JSON, and a verification script.
- `dsv4_sglang_geomean/`: DeepSeek-V4-Pro FP4 SGLang MI355X geomean bundle across seven TP8 non-disaggregated configs, with final exact geomean `3.4739x` / `247.4%` improvement.
- `kimi_fp4_geomean/`: Kimi-K2.5 MXFP4 vLLM MI355X geomean bundle across seven TP8 non-disaggregated configs (`8k1k` CONC 4/8/16/32/64, `1k1k` CONC 4/8), with final exact geomean `9.9584x` / `895.8%` improvement. Memory tuned to `gpu-memory-utilization 0.80` + `expandable_segments` for shared-GPU headroom.
- `glm5_fp8_atom_geomean/`: GLM-5 FP8 **ATOM** MI355X geomean bundle across seven TP8 non-disaggregated configs (`1k1k` CONC 1/2/4/8/16, `8k1k` CONC 4/8), with final exact geomean `3.1415x` / `214.1%` improvement. NEW variant uses the ATOM inference engine (`rocm/atom:...atom0.1.2.post`); OLD variant reuses the SGLang v0.5.8 baseline from `glm5_fp8_geomean/`. This is the headline GLM-5 result.
- `glm5_fp8_geomean/`: GLM-5 FP8 SGLang MI355X geomean bundle across seven TP8 non-disaggregated configs (`1k1k` CONC 1/2/4/8/16, `8k1k` CONC 4/8), with geomean `2.5290x` / `152.9%` improvement (SGLang old->new). Kept as the SGLang reference and as the shared OLD baseline for the ATOM bundle.
- `kimi_int4_8k1k_conc16_prompts3x/`: completed Kimi-K2.5 INT4 vLLM same-row local reproduction attempt for the previously shortlisted 8k/1k middle-state candidate. It completed successfully but did not reproduce the recorded 4x dashboard gain. The Kimi HF cache is backed up with hardlinks under `/scratch/model_backups/hf_hub_cache/models--moonshotai--Kimi-K2.5`.
- `kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x/`: Kimi-K2.5 MXFP4 old-image flag ablation for the dashboard candidate `348.532 -> 1647.260 tok/s/GPU`. Full 03-26 flags do not run on the old `v0.16.0` image at TP8 because AITER MLA rejects 8 local heads; the compatible subset without `block-size=1` completed at `290.762867 tok/s/GPU`.
- `mi355_adsuite_prompts3x/`: original broader MI355X suite output directory for Kimi-K2.5, GLM-5, and the failed DeepSeek-V4-Pro vLLM old-row attempt. This is also copied into the primary bundle under `dsv4_sglang_8k1k_conc8_prompts3x/suites/mi355_adsuite_prompts3x/`.
- `HANDOFF.md`: restart note for the next agent after this machine is released. It assumes only this git repo survives and documents what must be re-downloaded.
- `backup_inventory/`: small, git-safe inventories for model cache directories, Docker images, repo files, and one legacy root script. It does not contain model weights or Docker layers.

## Multi-Config Geomean Details

Five models, each swept across multiple non-disaggregated configs (`num_prompts = CONC * 3`; TP8 except DeepSeek-R1-0528 at TP4 and gpt-oss-120b at TP1 — `tok/s/GPU` is per-GPU normalized so all are comparable). The headline geomeans are in the summary table at the top; exact per-config rows are in each bundle's `comparison.csv`.

The Kimi-K2.5 headline geomean uses the three high-concurrency configs (`8k1k` CONC 64/128/256), where both old and new images are near saturation and the gain is a credible `3.71x`. The bundle also contains lower-concurrency configs (CONC 4-32), but at low concurrency the old image (vLLM v0.16.0) is pathologically slow, inflating per-config gains to 10-18x; those are kept for completeness but are not the reported headline.

The DeepSeek-R1-0528 headline geomean uses the three high-concurrency configs (`8k1k` CONC 32/64/128 = `2.31x / 2.77x / 2.60x`, geomean `2.55x`) at TP4. This is an end-to-end latest-vs-earliest comparison: the OLD end runs the MXFP4 checkpoint the v0.5.2 workflow shipped with (revision `6c94c74`, pre-2025-09-30) on `v0.5.2-rocm7.0` with the old server flags, while the NEW end runs the current HEAD checkpoint (revision `913fc83b`, which AMD further optimized across releases) on `v0.5.13`, so the gain reflects both the software/runtime upgrade and the model-level weight optimization done over time. `tput_per_gpu` is per-GPU normalized, so TP4 stays directly comparable to the other TP8 bundles. The prior pure-software, same-checkpoint TP8 number (`2.07x`) is retained in `dsr1_fp4_geomean/`.

The gpt-oss-120b geomean is a full 5-config sweep (`8k1k` CONC 16/32/64/128/256 = `1.06x / 1.16x / 1.14x / 1.12x / 1.29x`, geomean `1.15x`) on vLLM at TP1 (native MXFP4, ~65GB, fits one GPU). This is a modest but honest result: the old->new vLLM upgrade (`vllm_0.10.1` -> `vllm-openai-rocm:v0.22.0`, adding `ROCM_AITER_UNIFIED_ATTN`, `fuse_rope_kvcache`, INT4 quick-reduce) lifts throughput ~15% overall and up to `1.29x` at the highest concurrency, but gpt-oss lacks the DeepSeek-specific MoE/MLA kernel work that drives the 2-4x bundles. Kept as a complete record, not a headline number. Absolute throughput is high (`16k -> 49k tok/s/GPU`) because the 120B MoE activates only ~5B params per token.

## Top Inference Optimizations

The old->new gains above come from image/runtime upgrades shipped across many PRs in the `SemiAnalysisAI/InferenceX` perf changelog. The three biggest inference-side levers:

| # | Optimization | What it does | Models |
|---|---|---|---|
| 1 | AITER optimized kernels | Fused & optimized GPU kernels for inference (MHC, fused hash-topk, fused compress) | Kimi, DSV4 |
| 2 | MoE/Attention backend upgrades | FlyDSL MoE + Unified KV attention backends optimizations | DSV4, GLM-5 |
| 3 | Parallelism & scheduling tuning | Expert parallelism (EP), DP-attention, and two-batch overlap (TBO) | DSV4, Kimi |

Details:

1. **AITER optimized kernels.** Fused & optimized GPU kernels for inference — AITER MHC (hash-correction) pre/post, fused hash-topk, and fused compress. This is the dominant lever for Kimi (the v0.16->v0.18 jump in PR #936) and a recurring one for DeepSeek-V4-Pro (PR #1272 fused compress, #1300 AITER MHC pre/post, #1355 fused hash-topk).

2. **MoE/Attention backend upgrades.** FlyDSL MoE and the Triton attention backend (DeepSeek-V4-Pro PR #1355) plus NSA kernels (GLM-5 PR #762) replace the generic MoE/Attention paths with hardware-tuned ones on MI355X.

3. **Parallelism & scheduling tuning.** DeepSeek-V4-Pro and Kimi use expert parallelism (EP), DP-attention, and two-batch overlap (TBO) to keep all eight GPUs busy at high concurrency, combined with CONC-driven `--cuda-graph-max-bs` / `--max-running-requests` so graph-capture and serving capacity match each sweep point, plus KV-pool sizing to free memory for larger micro-batches.

<sub>Evidence in the InferenceX perf changelog: CONC-driven graph/serving sizing — DSV4 PR #1272; high-concurrency KV-pool fix — #1568; EP=8 + DP-attention + DeepEP recipe — #1868; MORI EP two-batch-overlapping / TBO — #783, #2150, and the DSV4 ATOM high-concurrency TBO entries. FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) shipped for GLM-5 in #1023.</sub>

Per-model attribution: Kimi's speedup is concentrated in the single v0.16->v0.18 jump (PR #936); later image bumps (v0.18->0.21->0.22) are version-only with no new runtime tuning. DeepSeek-V4-Pro accrues gains incrementally — nearly every image bump ships additional runtime-env tuning. GLM-5 on SGLang gains `2.53x`; switching the serving engine to ATOM (AMD's optimized inference engine, PR #1126) lifts it to `3.14x` against the same SGLang baseline, which is the headline GLM-5 number.

## Main Results

| Scope | Seq | CONC | Image | Result |
|---|---|---:|---|---|
| DeepSeek-V4-Pro FP4 SGLang geomean | mixed: `8k1k_c8`, `1k1k_c1/c2/c3/c4/c8`, `1k512_c1` | mixed | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | geomean `3.4739x`, improvement `247.4%`; exact rows in `dsv4_sglang_geomean/comparison.csv` |
| Kimi-K2.5 FP4 vLLM geomean | `8k1k_c64/c128/c256` | 64/128/256 | `vllm/vllm-openai-rocm:v0.16.0` -> `vllm/vllm-openai-rocm:v0.22.0` | geomean `3.7100x`, improvement `271.0%`; per-config `4.17x / 3.97x / 3.08x`; full 9-config sweep (incl. low-conc 10-18x outliers) in `kimi_fp4_geomean/comparison.csv` |
| GLM-5 FP8 ATOM geomean | mixed: `1k1k_c1/c2/c4/c8/c16`, `8k1k_c4/c8` | mixed | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` (SGLang old) -> `rocm/atom:...atom0.1.2.post` (ATOM new) | geomean `3.1415x`, improvement `214.1%`; per-config 2.98-3.38x; exact rows in `glm5_fp8_atom_geomean/comparison.csv` |
| GLM-5 FP8 SGLang geomean | mixed: `1k1k_c1/c2/c4/c8/c16`, `8k1k_c4/c8` | mixed | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` -> `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413` | geomean `2.5290x`, improvement `152.9%` (SGLang reference); exact rows in `glm5_fp8_geomean/comparison.csv` |
| DeepSeek-R1-0528 FP4 SGLang geomean (TP4, latest-vs-earliest ckpt) | `8k1k_c32/c64/c128` | 32/64/128 | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` + ckpt `6c94c74` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` + ckpt `913fc83b` (HEAD) | geomean `2.5547x`, improvement `155.5%`; per-config `2.31x / 2.77x / 2.60x`; exact rows in `dsr1_fp4_tp4_geomean/comparison.csv` |
| DeepSeek-R1-0528 FP4 SGLang geomean (TP8, same-ckpt pure-software) | `8k1k_c32/c64/c128` | 32/64/128 | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | geomean `2.0711x`, improvement `107.1%`; per-config `1.99x / 2.07x / 2.15x`; full 12-config sweep (all-config geomean `1.77x`) in `dsr1_fp4_geomean/comparison.csv` |
| gpt-oss-120b FP4 vLLM geomean (TP1) | `8k1k_c16/32/64/128/256` | 16/32/64/128/256 | `rocm/7.0:...vllm_0.10.1_instinct_20250927_rc1` -> `vllm/vllm-openai-rocm:v0.22.0` | geomean `1.1507x`, improvement `15.1%`; per-config `1.06x / 1.16x / 1.14x / 1.12x / 1.29x`; exact rows in `gptoss_120b_fp4_geomean/comparison.csv` |
| DeepSeek-V4-Pro FP4 SGLang | 8192/1024 | 8 | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | `104.044135 -> 421.982767 tok/s/GPU`, `4.0558x` |
| Kimi-K2.5 INT4 vLLM attempted middle-state | 8192/1024 | 16 | `vllm/vllm-openai-rocm:v0.15.1` -> `vllm/vllm-openai-rocm:v0.18.0` | local `142.174440 -> 187.669496 tok/s/GPU`, `1.3200x`; did not reproduce the recorded `4.0143x` dashboard candidate |
| Kimi-K2.5 FP4 vLLM | 8192/1024 | 4 | `vllm/vllm-openai-rocm:v0.16.0` -> `vllm/vllm-openai-rocm:v0.22.0` | `24.517507 -> 359.122629 tok/s/GPU`, `14.6476x` |
| GLM-5 FP8 SGLang | 1024/1024 | 4 | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` -> `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413` | `16.169320 -> 41.515338 tok/s/GPU`, `2.5675x` |
| DeepSeek-V4-Pro FP4 vLLM | 1024/1024 | 4 | original old image unavailable; substitute `vllm/vllm-openai-rocm:v0.21.0` -> `vllm/vllm-openai-rocm:v0.22.0` | old-row local repro failed because the original workflow image was unavailable and the substitute image failed before benchmark |
| Qwen3-235B-A22B FP8 SGLang attempted (TP4) | 8192/1024 | 64 | `rocm/7.0:...sgl-dev-v0.5.2-rocm7.0-mi35x-20250915` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` (same two SGLang images as DeepSeek-R1) | `618.68 -> 627.90 tok/s/GPU`, `1.01x`; essentially no gain (TTFT 14.5s->11.2s but decode TPOT flat). Model `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8`. Qwen3-MoE lacks the DeepSeek-specific MoE/MLA kernels, so the old->new image upgrade barely moves throughput; no bundle kept |

## Kimi Middle-State Candidates

The existing Kimi FP4 `CONC=4` result is preserved because it is valid, but the gain is large (`14.6476x` local, `13.05x` dashboard). A separate candidate scan is saved in:

```text
dsv4_sglang_8k1k_conc8_prompts3x/docs/candidates/kimi_middle_state_candidates.md
```

Important erratum after local repro:

| Scope | Seq | CONC | Status |
|---|---|---:|---|
| Kimi-K2.5 INT4 vLLM | 8192/1024 | 16 | completed locally as `142.174 -> 187.669 tok/s/GPU`, `1.320x`; do not use this as the 4x Kimi story |

The cleaner same-workflow Kimi INT4 3.5x-4.5x dashboard candidate is:

| Scope | Seq | CONC | Result |
|---|---|---:|---|
| Kimi-K2.5 INT4 vLLM | 1024/8192 | 16 | dashboard `22.890 -> 95.645 tok/s/GPU`, `4.178x`; not locally reproduced yet |

If the story must stay strictly Kimi FP4, there is no same-config 3.5x-4.5x pair in the current MI355X history. The closest FP4 candidate is:

| Scope | Seq | CONC | Result |
|---|---|---:|---|
| Kimi-K2.5 FP4 vLLM | 8192/1024 | 64 | `348.532 -> 1647.260 tok/s/GPU`, `4.726x` |

Old-image flag ablation for this FP4 candidate is saved in `kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x/`. The fair full-new-flags-on-old-image run is blocked: `VLLM_ROCM_USE_AITER=1` and `--block-size=1` each trigger the old `v0.16.0` AITER MLA assertion at TP8 (`Provided 8 number of heads`). Keeping `block_size=64` while applying the other compatible server flags completed locally at `290.762867 tok/s/GPU` with `CONC=64` and `num_prompts=192`.

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

The **ROCm software stack that runs the benchmarks lives inside each Docker image** (e.g. the SGLang/vLLM/ATOM ROCm images pinned per bundle), so no host ROCm change is required — only a working amdgpu/KFD driver and Docker GPU access. The host ROCm version above is informational.

Per-bundle Docker images and the ROCm version each ships (old = baseline, new = optimized):

| Bundle | Variant | Image | ROCm in image |
|---|---|---|---|
| DSV4 | old | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` | 7.2.x (per tag) |
| DSV4 | new | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 7.2.x (per tag) |
| Kimi FP4 | old | `vllm/vllm-openai-rocm:v0.16.0` | 7.0.0 |
| Kimi FP4 | new | `vllm/vllm-openai-rocm:v0.22.0` | 7.2.2 |
| GLM-5 | old | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` | 7.2.0 |
| GLM-5 ATOM | new | `rocm/atom:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.2.post` | 7.2.2 |

(The SGLang-new GLM image `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413`, used by the reference `glm5_fp8_geomean` bundle, ships ROCm 7.2.0.)

**0. Prerequisites**
- 8× AMD Instinct MI355X, ROCm 7.2.x, Docker with GPU access (`--device=/dev/kfd --device=/dev/dri`).
- A Hugging Face token with access to the model repos below.
- Enough fast local scratch for one model at a time (weights are 0.5–0.8 TB each). Pick any directory; the scripts take it via env vars — nothing is hardcoded to a fixed install path.

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
| DSV4 | `deepseek-ai/DeepSeek-V4-Pro` | `5607980f3a4b8ea0371b9f11e1848ac41f14979e` |
| Kimi FP4 | `amd/Kimi-K2.5-MXFP4` | `419004c8716cf22c929aa15d39b85e09a8a2091a` |
| GLM-5 | `zai-org/GLM-5-FP8` | `4f96cc5eec29dcee5d6ded54f7ffe889438f9516` |
| DSR1 FP4 (OLD end) | `amd/DeepSeek-R1-0528-MXFP4-Preview` | `6c94c74df15d3eabd6cbc71bb12a2b31dae6f5ff` |
| DSR1 FP4 (NEW end, HEAD) | `amd/DeepSeek-R1-0528-MXFP4-Preview` | `913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a` |
| gpt-oss-120b | `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |

```bash
HF_HUB_CACHE=$MODEL_DIR HF_HOME=$HF_HOME_DIR python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amd/Kimi-K2.5-MXFP4", revision="419004c8716cf22c929aa15d39b85e09a8a2091a")
PY
```

**3. Run the repro sweep** (each script pulls its old/new Docker images automatically, patches the benchmark for `num_prompts = CONC*3`, and writes results into the bundle):

```bash
# DeepSeek-V4-Pro FP4 SGLang -> geomean 3.47x
cd "$REPRO_ROOT/dsv4_sglang_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD TARGETS=all scripts/repro_dsv4_sglang_geomean.sh

# Kimi-K2.5 FP4 vLLM -> high-conc geomean 3.71x
cd "$REPRO_ROOT/kimi_fp4_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD TARGETS=all scripts/repro_kimi_fp4_geomean.sh

# GLM-5 FP8 ATOM (new) vs SGLang baseline (old) -> geomean 3.14x.
# First run the SGLang baseline (old) once so ATOM can reuse it:
cd "$REPRO_ROOT/glm5_fp8_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD TARGETS=old scripts/repro_glm5_fp8_geomean.sh
cd "$REPRO_ROOT/glm5_fp8_atom_geomean"
REPO=/root/InferenceX MODEL_DIR=$MODEL_DIR HF_HOME_DIR=$HF_HOME_DIR WT_ROOT=$WT_ROOT \
  OUT_DIR=$PWD SGLANG_OLD_DIR="$REPRO_ROOT/glm5_fp8_geomean" TARGETS=new scripts/repro_glm5_fp8_atom_geomean.sh
```

Notes: on a shared machine the scripts already lower memory (`GPU_MEM_UTIL=0.80`) and use port `19000`; override with `GPU_MEM_UTIL=` / `PORT=` if needed. `SKIP_EXISTING=1` (default) lets you resume without re-running finished configs.

**4. Verify** the resulting numbers with the scripts in the next section.

## Verify

All commands below are relative to the repository root — run them from wherever you cloned this repo (no fixed install path required).

Primary handoff bundle:

```bash
cd dsv4_sglang_8k1k_conc8_prompts3x
scripts/verify_bundle.sh
```

Expected: all checks pass, including checksum validation, HF token scan, cache restore verification, DSV4 SGLang numeric results, and Kimi/GLM suite numeric results.

DSV4 SGLang geomean bundle:

```bash
cd dsv4_sglang_geomean
scripts/verify_dsv4_sglang_geomean.sh
```

Expected: `ok: 7 configs, geomean=3.473867x`.

Kimi FP4 geomean bundle:

```bash
cd kimi_fp4_geomean
scripts/verify_kimi_fp4_geomean.sh
```

Expected: `ok: 9 configs; high-conc (c64/c128/c256) geomean=3.709068x`.

GLM-5 ATOM geomean bundle:

```bash
cd glm5_fp8_atom_geomean
scripts/verify_glm5_fp8_atom_geomean.sh
```

Expected: `ok: 7 configs, ATOM-vs-SGLang geomean=3.141457x`.

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

gpt-oss-120b FP4 vLLM geomean bundle:

```bash
cd gptoss_120b_fp4_geomean
scripts/verify_gptoss_120b_fp4_geomean.sh
```

Expected: `ok: 5 configs (8k1k c16-c256, TP1) geomean=1.150664x; peak 8k1k_c256=1.286x`.

## Git Notes

This parent repo tracks the whole run tree as ordinary files; it can be cloned to any location (no fixed install path). The nested git metadata from `dsv4_sglang_8k1k_conc8_prompts3x/.git` was moved out before committing so the directory is tracked as ordinary files rather than as a submodule.

If this machine is released, use `HANDOFF.md` as the first file to read. The large local state under `/scratch/hf_hub_cache`, `/scratch/model_backups`, `/dev/shm`, and Docker storage is not preserved by this git repo.
