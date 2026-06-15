#!/usr/bin/env bash
set -euo pipefail

# Token is intentionally not written here. Load it at runtime:
# export HF_TOKEN="$(cat /root/.cache/huggingface/token)"

export REPO=/root/InferenceX
export WT_ROOT=/scratch/inferencex_worktrees/dsv4_sglang_8k1k_conc8
export OUT_DIR=/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
export MODEL_DIR=/scratch/hf_hub_cache
export HF_HOME_DIR=/scratch/hf_home
export PORT=18888
export PROMPT_MULTIPLIER=3
export RANDOM_RANGE_RATIO=0.8
export SKIP_EXISTING=1
export DRY_RUN=0
export CONTINUE_ON_ERROR=0

TARGETS=all /root/repro_dsv4_sglang_8k1k_conc8_tp8.sh
TARGETS=old /root/repro_dsv4_sglang_8k1k_conc8_tp8.sh
TARGETS=new /root/repro_dsv4_sglang_8k1k_conc8_tp8.sh
