# MI355X Repro Bundle

This directory is a self-contained handoff bundle for the local MI355X reproductions and candidate research we discussed. It contains:

- the primary strict same-config DeepSeek-V4-Pro FP4 SGLang old/new reproduction;
- the broader MI355X suite with Kimi-K2.5, GLM-5, and the failed DeepSeek-V4-Pro vLLM old-row attempt;
- the candidate tables used to decide which models were worth reproducing.

## Primary Claim

Compare the old and new InferenceX workflow rows with the same model, framework, precision, hardware, sequence length, concurrency, tensor parallelism, expert parallelism, and non-disaggregated mode:

| Variant | Workflow run | Attempt | Commit | Image | ISL/OSL | CONC | TP | EP | DP attention | Prompts |
|---|---:|---:|---|---|---|---:|---:|---:|---|---:|
| old | 25262278289 | 1 | `f93f91dfb0ef` | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` | 8192/1024 | 8 | 8 | 1 | false | 24 |
| new | 27475314602 | 3 | `8d206016537a` | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 8192/1024 | 8 | 8 | 1 | false | 24 |

The workflow default is `num_prompts = CONC * 10`; this local repro intentionally used `num_prompts = CONC * 3 = 24` to keep runtime shorter.

## Result

Both targets completed with `success` on 2026-06-15 UTC.

| Variant | Local tok/s/GPU | Local total tok/s | Local output tok/s/GPU | Mean TTFT ms | Mean TPOT ms | Dashboard tok/s/GPU |
|---|---:|---:|---:|---:|---:|---:|
| old | 104.044135 | 832.353076 | 11.774508 | 6143.49 | 75.4615 | 115.801807 |
| new | 421.982767 | 3375.862134 | 47.755112 | 1310.04 | 18.6352 | 483.774241 |

Measured strict same-config gain:

| Old tok/s/GPU | New tok/s/GPU | Gain | Improvement |
|---:|---:|---:|---:|
| 104.044135 | 421.982767 | 4.0558x | +305.58% |

Dashboard reference gain: `483.774241 / 115.801807 = 4.1776x`.

## Other Models

The earlier multi-model suite is preserved under `suites/mi355_adsuite_prompts3x/`. It used `num_prompts = CONC * 3`, `TP=8`, non-disaggregated, no spec decoding.

| Model | Precision | Framework | Seq | CONC | Old status | Old tok/s/GPU | New status | New tok/s/GPU | Measured gain | Dashboard reference |
|---|---|---|---|---:|---|---:|---|---:|---:|---:|
| Kimi-K2.5 | fp4 | vLLM | 8192/1024 | 4 | success | 24.517507 | success | 359.122629 | 14.6476x (+1364.76%) | 28.7 -> 374.2 |
| GLM-5 | fp8 | SGLang | 1024/1024 | 4 | success | 16.169320 | success | 41.515338 | 2.5675x (+156.75%) | 17.5 -> 44.2 |
| DeepSeek-V4-Pro | fp4 | vLLM | 1024/1024 | 4 | `docker_failed:1` | n/a | success | 21.328874 | n/a | 3.3 -> 23.3 |

The DeepSeek-V4-Pro vLLM old row is intentionally kept as a failed attempt. Its original workflow image tag was no longer available, so the suite used the closest public substitute (`vllm/vllm-openai-rocm:v0.21.0`) and a runtime PDL-disable patch; it still failed before benchmark with `Cannot find global function target.build.tilelang_hip`. The DSV4 SGLang row above is the clean replacement reproduction.

Candidate tables are included under `docs/candidates/`. They distinguish rows we merely shortlisted from rows actually run locally:

| Candidate model | Best MI355X non-disagg no-ATOM dashboard row | Local status in this bundle |
|---|---|---|
| DeepSeek-R1-0528 | 1,032.3 -> 3,658.6 tok/s/GPU, 3.54x | candidate only |
| DeepSeek-V4-Pro | 3.3 -> 23.3 vLLM dashboard row, 7.00x | vLLM old failed; SGLang replacement succeeded |
| Kimi-K2.5 | 28.7 -> 374.2 tok/s/GPU, 13.05x | reproduced successfully |
| MiniMax-M3 | 4.3 -> 15.5 tok/s/GPU, 3.62x | candidate only |
| GLM-5 | 17.5 -> 44.2 tok/s/GPU, 2.52x | reproduced successfully |
| MiniMax-M2.5 | 55.1 -> 98.2 tok/s/GPU, 1.78x | candidate only |
| Qwen-3.5-397B-A17B | 42.7 -> 107.5 tok/s/GPU, 2.52x | candidate only |

For Kimi-specific middle-state options around 3.5x-4.5x, see `docs/candidates/kimi_middle_state_candidates.md`. The short version: strict Kimi FP4 has no same-config 3.5x-4.5x pair; the best exact-band candidate is Kimi `int4`, `8192/1024`, `CONC=16`, at `4.014x`, while the nearest FP4 candidate is `8192/1024`, `CONC=64`, at `4.726x`.

## Bundle Layout

- `comparison.csv`: one-row old/new comparison and gain.
- `combined_results.csv`: normalized per-run result table.
- `runs/dsv4_sglang_old/`: old raw JSON, aggregate JSON, logs, GPU metrics, env, and local patch.
- `runs/dsv4_sglang_new/`: new raw JSON, aggregate JSON, logs, GPU metrics, env, and local patch.
- `cache_status/`: image manifest checks and HF cache mutation/restore records.
- `workflow_metadata/`: GitHub Actions run/job metadata captured during research.
- `scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh`: exact local runner used for this bundle.
- `scripts/repro_mi355_adsuite.sh`: runner used for the Kimi/GLM/DSV4-vLLM broader suite.
- `scripts/verify_bundle.sh`: one-command acceptance check for this bundle.
- `docs/dsv4_sglang_8k1k_conc8_repro_plan.md`: research handoff with source row details.
- `docs/candidates/`: MI355X no-disagg no-ATOM candidate tables.
- `suites/mi355_adsuite_prompts3x/`: prior multi-model local reproduction suite.
- `MANIFEST.sha256`: checksum manifest for the committed bundle contents.

## Reproduce

Prereqs on this host:

- `/root/InferenceX` exists and can fetch the two commits.
- Docker can run MI355/MI35X ROCm containers.
- The DeepSeek-V4-Pro cache exists at `/scratch/hf_hub_cache`.
- HF token is available via `/root/.cache/huggingface/token` or `HF_TOKEN`.
- Port `18888` is free, or set `PORT` to another free port.

Run both old and new:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
TARGETS=all scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh
```

Run only one side:

```bash
TARGETS=old scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh
TARGETS=new scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh
```

Useful knobs:

```bash
export OUT_DIR=/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
export WT_ROOT=/scratch/inferencex_worktrees/dsv4_sglang_8k1k_conc8
export MODEL_DIR=/scratch/hf_hub_cache
export HF_HOME_DIR=/scratch/hf_home
export PROMPT_MULTIPLIER=3
export PORT=18888
```

The old SGLang script mutates the cached HF `config.json` from `deepseek_v4` to `deepseek_v3`. The runner backs up the real blob target before the old run and restores it afterward. Cache status is recorded under `cache_status/`.

Run the broader suite:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
OUT_DIR="$PWD/suites/mi355_adsuite_prompts3x" TARGETS=all scripts/repro_mi355_adsuite.sh
```

Run only part of the broader suite:

```bash
OUT_DIR="$PWD/suites/mi355_adsuite_prompts3x" TARGETS=kimi scripts/repro_mi355_adsuite.sh
OUT_DIR="$PWD/suites/mi355_adsuite_prompts3x" TARGETS=glm scripts/repro_mi355_adsuite.sh
OUT_DIR="$PWD/suites/mi355_adsuite_prompts3x" TARGETS=dsv4 scripts/repro_mi355_adsuite.sh
```

## Acceptance

Validate the saved bundle:

```bash
cd /scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
scripts/verify_bundle.sh
```

The verification checks:

- all required result, log, metadata, and script files are present;
- old and new status files are `success`;
- `MANIFEST.sha256` matches the current bundle files;
- no persisted HF token matching `hf_[A-Za-z0-9]{20,}` exists in the bundle;
- HF cache initial/final records are both `model_type=deepseek_v4` with unchanged sha256;
- `comparison.csv` contains `104.04413451994634 -> 421.9827666910809`, `4.0558054390868366x`, and `+305.58054390868364%`;
- aggregate JSON metadata matches MI355X, SGLang, FP4, `8192/1024`, `CONC=8`, `TP=8`.
- the multi-model suite contains Kimi and GLM success results and the known DSV4 vLLM old-row `docker_failed:1` result;
- Kimi local result is `24.517506602596864 -> 359.12262861606865 tok/s/GPU`;
- GLM local result is `16.169319874259486 -> 41.51533759847504 tok/s/GPU`.

Manual spot checks:

```bash
cat comparison.csv
cat suites/mi355_adsuite_prompts3x/comparison.csv
cat cache_status/initial.txt
cat cache_status/final.txt
git status --short
```

Expected `git status --short` after the bundle commit is clean.
