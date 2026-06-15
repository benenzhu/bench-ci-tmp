# InferenceX MI355X Repro Runs

This repository contains the local MI355X benchmark reproduction artifacts produced on 2026-06-15 UTC.

## Contents

- `dsv4_sglang_8k1k_conc8_prompts3x/`: primary handoff bundle. It includes the strict same-config DeepSeek-V4-Pro FP4 SGLang reproduction, the broader MI355X Kimi/GLM suite copied under `suites/`, candidate tables, logs, raw JSON, aggregate JSON, and a verification script.
- `mi355_adsuite_prompts3x/`: original broader MI355X suite output directory for Kimi-K2.5, GLM-5, and the failed DeepSeek-V4-Pro vLLM old-row attempt. This is also copied into the primary bundle under `dsv4_sglang_8k1k_conc8_prompts3x/suites/mi355_adsuite_prompts3x/`.

## Main Results

| Scope | Seq | CONC | Result |
|---|---|---:|---|
| DeepSeek-V4-Pro FP4 SGLang | 8192/1024 | 8 | `104.044135 -> 421.982767 tok/s/GPU`, `4.0558x` |
| Kimi-K2.5 FP4 vLLM | 8192/1024 | 4 | `24.517507 -> 359.122629 tok/s/GPU`, `14.6476x` |
| GLM-5 FP8 SGLang | 1024/1024 | 4 | `16.169320 -> 41.515338 tok/s/GPU`, `2.5675x` |
| DeepSeek-V4-Pro FP4 vLLM | 1024/1024 | 4 | old-row local repro failed because the original workflow image was unavailable and the substitute image failed before benchmark |

## Verify

Run:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
scripts/verify_bundle.sh
```

Expected: all checks pass, including checksum validation, HF token scan, cache restore verification, DSV4 SGLang numeric results, and Kimi/GLM suite numeric results.

## Git Notes

This parent repo is intended for pushing the whole `/scratch/inferencex_runs` tree. The nested git metadata from `dsv4_sglang_8k1k_conc8_prompts3x/.git` was moved out before committing so the directory is tracked as ordinary files rather than as a submodule.
