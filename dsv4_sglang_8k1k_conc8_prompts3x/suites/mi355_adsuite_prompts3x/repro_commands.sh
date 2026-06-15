#!/usr/bin/env bash
set -euo pipefail

# Token is intentionally not written here. Load it at runtime:
# export HF_TOKEN="$(cat /root/.cache/huggingface/token)"

export REPO=/root/InferenceX
export WT_ROOT=/scratch/inferencex_worktrees/mi355_adsuite_repro
export OUT_DIR=/scratch/inferencex_runs/mi355_adsuite_prompts3x
export MODEL_DIR=/scratch/hf_hub_cache
export HF_HOME_DIR=/scratch/hf_home
export BACKUP_CACHE_DIR=/scratch/hf_hub_cache
export PORT=18888
export PROMPT_MULTIPLIER=3
export RANDOM_RANGE_RATIO=0.8
export SKIP_EXISTING=1
export DRY_RUN=0
export CONTINUE_ON_ERROR=0

# Full suite:
TARGETS=all /root/repro_mi355_adsuite.sh

# Individual model pairs:
TARGETS=kimi /root/repro_mi355_adsuite.sh
TARGETS=dsv4 /root/repro_mi355_adsuite.sh
TARGETS=glm /root/repro_mi355_adsuite.sh

# Individual labels:
TARGETS=kimi_old /root/repro_mi355_adsuite.sh
TARGETS=kimi_new /root/repro_mi355_adsuite.sh
TARGETS=dsv4_old /root/repro_mi355_adsuite.sh
TARGETS=dsv4_new /root/repro_mi355_adsuite.sh
TARGETS=glm_old /root/repro_mi355_adsuite.sh
TARGETS=glm_new /root/repro_mi355_adsuite.sh
