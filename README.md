# InferenceX MI355X Repro Runs

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

## Multi-Config Geomean Summary

Three models, each swept across multiple TP8 non-disaggregated configs (`num_prompts = CONC * 3`), reporting the geometric mean of per-config old->new `tok/s/GPU` gains:

| Model | Prec / Framework | Configs | Geomean | Improvement | Bundle |
|---|---|---:|---:|---:|---|
| DeepSeek-V4-Pro | FP4 / SGLang | 7 | `3.4739x` | `247.4%` | `dsv4_sglang_geomean/` |
| Kimi-K2.5 | FP4 / vLLM | 3 | `3.7100x` | `271.0%` | `kimi_fp4_geomean/` |
| GLM-5 | FP8 / SGLang→ATOM | 7 | `3.1415x` | `214.1%` | `glm5_fp8_atom_geomean/` |

Exact per-config rows are in each bundle's `comparison.csv`.

The Kimi-K2.5 headline geomean uses the three high-concurrency configs (`8k1k` CONC 64/128/256), where both old and new images are near saturation and the gain is a credible `3.71x`. The bundle also contains lower-concurrency configs (CONC 4-32), but at low concurrency the old image (vLLM v0.16.0) is pathologically slow, inflating per-config gains to 10-18x; those are kept for completeness but are not the reported headline.

## Top Inference Optimizations

The old->new gains above come from image/runtime upgrades shipped across many PRs in the `SemiAnalysisAI/InferenceX` perf changelog. The three biggest inference-side levers:

| # | Optimization | What it does | Models |
|---|---|---|---|
| 1 | AITER optimized kernels | Fused & optimized GPU kernels for inference (MHC, fused hash-topk, fused compress) | Kimi, DSV4 |
| 2 | MoE/Attention backend upgrades | FlyDSL MoE + Triton attention backend and NSA kernels optimizations | DSV4, GLM-5 |
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
| DeepSeek-V4-Pro FP4 SGLang | 8192/1024 | 8 | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` -> `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | `104.044135 -> 421.982767 tok/s/GPU`, `4.0558x` |
| Kimi-K2.5 INT4 vLLM attempted middle-state | 8192/1024 | 16 | `vllm/vllm-openai-rocm:v0.15.1` -> `vllm/vllm-openai-rocm:v0.18.0` | local `142.174440 -> 187.669496 tok/s/GPU`, `1.3200x`; did not reproduce the recorded `4.0143x` dashboard candidate |
| Kimi-K2.5 FP4 vLLM | 8192/1024 | 4 | `vllm/vllm-openai-rocm:v0.16.0` -> `vllm/vllm-openai-rocm:v0.22.0` | `24.517507 -> 359.122629 tok/s/GPU`, `14.6476x` |
| GLM-5 FP8 SGLang | 1024/1024 | 4 | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` -> `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413` | `16.169320 -> 41.515338 tok/s/GPU`, `2.5675x` |
| DeepSeek-V4-Pro FP4 vLLM | 1024/1024 | 4 | original old image unavailable; substitute `vllm/vllm-openai-rocm:v0.21.0` -> `vllm/vllm-openai-rocm:v0.22.0` | old-row local repro failed because the original workflow image was unavailable and the substitute image failed before benchmark |

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

## Verify

Run:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
scripts/verify_bundle.sh
```

Expected: all checks pass, including checksum validation, HF token scan, cache restore verification, DSV4 SGLang numeric results, and Kimi/GLM suite numeric results.

For the Kimi INT4 local attempt:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
scripts/verify_kimi_int4_8k1k_conc16.sh
```

Expected: all checks pass, and the script prints that the local attempt completed but did not reproduce the recorded `4.0143x` dashboard gain.

For the DSV4 SGLang geomean bundle:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_geomean
scripts/verify_dsv4_sglang_geomean.sh
```

Expected: `ok: 7 configs, geomean=3.473867x`.

## Git Notes

This parent repo is intended for pushing the whole `/scratch/inferencex_runs` tree. The nested git metadata from `dsv4_sglang_8k1k_conc8_prompts3x/.git` was moved out before committing so the directory is tracked as ordinary files rather than as a submodule.

If this machine is released, use `HANDOFF.md` as the first file to read. The large local state under `/scratch/hf_hub_cache`, `/scratch/model_backups`, `/dev/shm`, and Docker storage is not preserved by this git repo.
