# Kimi-K2.5 INT4 vLLM MI355X Repro

Generated: 2026-06-15T08:05:57Z

Scope:

- Hardware: MI355X
- Model: `moonshotai/Kimi-K2.5`
- Precision/framework: INT4 / vLLM
- Mode: single-node non-disaggregated
- Sequence/concurrency: ISL/OSL `8192/1024`, CONC `16`
- Parallelism: TP `8`, EP `1`, DP attention `false`
- Spec decoding: none
- Prompt count patch: `num_prompts = CONC * 3`
- Local num prompts: `48`
- HF cache: `/scratch/hf_hub_cache`
- HF home/xet cache: `/scratch/hf_home`
- Output root: `/scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x`

Targets:

| label | run | attempt | job | commit | image | script | dashboard tok/s/GPU |
|---|---:|---:|---:|---|---|---|---:|
| kimi_int4_old | 22129277189 | 3 | 64160961756 | `d4f6998d1e5a` | `vllm/vllm-openai-rocm:v0.15.1` | `benchmarks/kimik2.5_int4_mi355x.sh` | 157.195420 |
| kimi_int4_new | 23669977901 | 6 | 70014502875 | `79a42e20a855` | `vllm/vllm-openai-rocm:v0.18.0` | `benchmarks/single_node/kimik2.5_int4_mi355x.sh` | 631.025997 |

Run:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
TARGETS=all scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
```

## Local Repro Result

| Old status | New status | Old tok/s/GPU | New tok/s/GPU | Gain | Improvement |
|---|---|---:|---:|---:|---:|
| success | success | 142.174440 | 187.669496 | 1.3200x | +32.00% |

Dashboard reference: `157.195420 -> 631.025997 tok/s/GPU`, `4.0143x`.

## Outcome

This is a completed local reproduction attempt, but it did **not** reproduce the
`4.0143x` dashboard claim for the requested `8192/1024`, `CONC=16`, `TP=8`
Kimi INT4 row. Treat this bundle as evidence that the previously shortlisted
8k/1k Kimi INT4 middle-state candidate is not usable without more investigation.

Both workflow job IDs match the requested single-node MI355X row:

- old: `22129277189/attempts/3`, job `64160961756`, image `vllm/vllm-openai-rocm:v0.15.1`
- new: `23669977901/attempts/6`, job `70014502875`, image `vllm/vllm-openai-rocm:v0.18.0`

The raw benchmark ran the same 48 prompts on both sides:

| Variant | Completed | Input tokens | Output tokens | Duration s | Total tok/s | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| old | 48 | 351295 | 44318 | 347.824 | 1137.396 | 127.415 |
| new | 48 | 351295 | 44318 | 263.504 | 1501.356 | 168.187 |

## Follow-Up Candidate

The same workflow pair still has a dashboard candidate in the desired
3.5x-4.5x band, but it is `1024/8192`, not this `8192/1024` local run:

| Precision | Seq | CONC | TP | Dashboard tok/s/GPU | Gain | Runs |
|---|---|---:|---:|---:|---:|---|
| int4 | 1024/8192 | 16 | 8 | `22.890 -> 95.645` | `4.178x` | `22129277189/attempts/3` -> `23669977901/attempts/6` |

That `1024/8192` candidate has not been run locally in this bundle.

## Model Cache Backup

The Kimi model cache is on the 40T `/scratch` filesystem:

- source: `/scratch/hf_hub_cache/models--moonshotai--Kimi-K2.5`
- hardlink backup: `/scratch/model_backups/hf_hub_cache/models--moonshotai--Kimi-K2.5`
- apparent size: `555G`
- regular files: `94`
- backup method: `cp -al`, so restore/deletion protection exists without
  duplicating the 555G of blob data on disk

Backup verification is saved in `cache_status/model_backup.txt`. A sample blob
had the same inode in source and backup, with link count `2`.

Restore from the backup:

```bash
mkdir -p /scratch/hf_hub_cache/models--moonshotai--Kimi-K2.5
rsync -aH --info=progress2 \
  /scratch/model_backups/hf_hub_cache/models--moonshotai--Kimi-K2.5/ \
  /scratch/hf_hub_cache/models--moonshotai--Kimi-K2.5/
```

## Reproduce

The HF token is intentionally not persisted in this directory. Use the existing
host token file or set `HF_TOKEN` before running:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
TARGETS=all scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
```

Run only one side:

```bash
TARGETS=old scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
TARGETS=new scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
```

Useful knobs:

```bash
export OUT_DIR=/scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
export WT_ROOT=/scratch/inferencex_worktrees/kimi_int4_8k1k_conc16
export MODEL_DIR=/scratch/hf_hub_cache
export HF_HOME_DIR=/scratch/hf_home
export PROMPT_MULTIPLIER=3
export PORT=18888
```

## Acceptance

Validate the saved bundle:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
scripts/verify_kimi_int4_8k1k_conc16.sh
```

Expected result: required logs/metadata/scripts exist, both status files are
`success`, no HF token is persisted, checksums match, and `comparison.csv`
contains `142.174440 -> 187.669496 tok/s/GPU`, `1.3200x`.
