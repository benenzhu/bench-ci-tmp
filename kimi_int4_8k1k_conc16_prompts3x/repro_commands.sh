#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Token is intentionally not written here. Load it at runtime:
# export HF_TOKEN="$(cat /root/.cache/huggingface/token)"

export REPO=/root/InferenceX
export WT_ROOT=/scratch/inferencex_worktrees/kimi_int4_8k1k_conc16
export OUT_DIR=/scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
export MODEL_DIR=/scratch/hf_hub_cache
export HF_HOME_DIR=/scratch/hf_home
export PORT=18888
export PROMPT_MULTIPLIER=3
export RANDOM_RANGE_RATIO=0.8
export SKIP_EXISTING=1
export DRY_RUN=0
export CONTINUE_ON_ERROR=0

TARGETS=all "$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
TARGETS=old "$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
TARGETS=new "$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
