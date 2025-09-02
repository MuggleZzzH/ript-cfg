#!/bin/bash

# ======================================================================
# Quick smoke test (fast) for PI0 + CFG-Flow
# - Minimal steps, tiny batch, short episodes
# - Uses Hydra overrides only; does NOT modify any config files
# ======================================================================

set -eo pipefail

# --- Locate project root ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
cd "$PROJECT_ROOT"
echo "[FAST TEST] Project root: $(pwd)"

# --- Optional: conda (comment out if not needed) ---
source /opt/conda/etc/profile.d/conda.sh && conda activate mix

# --- Env tweaks ---
export PI0_N_VIDEO=2
export PI0_VERBOSE=1
export PI0_VIDEO_DIR=output/videos_pi0
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT/LIBERO":"$PROJECT_ROOT"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PI0_DISABLE_INIT_STATES=0  # 快速测试：禁用初始状态设置

# --- Paths (edit if needed) ---
NORM_STATS=${NORM_STATS:-/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json}
PRETRAIN_PATH=${PRETRAIN_PATH:-/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch}

# --- Speed knobs (fast defaults) ---
TRAINING_STEPS=${TRAINING_STEPS:-2}             # total steps; cut=1 will stop after 1 loop
BATCH_SIZE=${BATCH_SIZE:-2}
RLOO_BATCH=${RLOO_BATCH:-2}
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-2}
NUM_ENVS=${NUM_ENVS:-2}
EARLY_STOP_PCT=${EARLY_STOP_PCT:-1}
ENABLE_DYNAMIC_SAMPLING=${ENABLE_DYNAMIC_SAMPLING:-false}
MAX_EP_LEN=${MAX_EP_LEN:-100}
WAIT_STEPS=${WAIT_STEPS:-10}
STRIDE=${STRIDE:-10}
MAX_WINDOWS=${MAX_WINDOWS:-5}
OPTIMIZER_BATCH=${OPTIMIZER_BATCH:-4}
CONDITION_MODE=${CONDITION_MODE:-token}
ROLLOUT_ENABLED=${ROLLOUT_ENABLED:-true}

echo "[FAST TEST] Params: steps=$TRAINING_STEPS, batch=$BATCH_SIZE, rloo=$RLOO_BATCH, rollouts_per_env=$ROLLOUTS_PER_ENV"
echo "[FAST TEST] Episode: max_len=$MAX_EP_LEN, wait=$WAIT_STEPS; windows: stride=$STRIDE, max=$MAX_WINDOWS"
echo "[FAST TEST] Dynamic sampling=$ENABLE_DYNAMIC_SAMPLING, early_stop=$EARLY_STOP_PCT, rollout.enabled=$ROLLOUT_ENABLED"

# --- Distributed single-process wiring ---
MASTER_PORT=$(python - << 'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT python train_ript_pi0.py \
  --config-name=train_base_rl_openvla_oft \
  algo=pi0_cfg_rl \
  algo.norm_stats_path=$NORM_STATS \
  algo.policy.pretrained_path=$PRETRAIN_PATH \
  training.n_steps=$TRAINING_STEPS \
  training.cut=1 \
  train_dataloader.batch_size=$BATCH_SIZE \
  algo.rloo_batch_size=$RLOO_BATCH \
  algo.rollouts_per_env=$ROLLOUTS_PER_ENV \
  algo.env_runner.num_parallel_envs=$NUM_ENVS \
  algo.early_stop_percentage=$EARLY_STOP_PCT \
  algo.enable_dynamic_sampling=$ENABLE_DYNAMIC_SAMPLING \
  algo.max_episode_length=$MAX_EP_LEN \
  algo.env_runner.num_steps_wait=$WAIT_STEPS \
  algo.stride=$STRIDE \
  algo.max_windows_per_episode=$MAX_WINDOWS \
  algo.rl_optimizer_factory.optimizer_batch_size=$OPTIMIZER_BATCH \
  algo.policy.condition_mode=$CONDITION_MODE \
  rollout.enabled=$ROLLOUT_ENABLED \
  logging.mode=disabled

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "[FAST TEST] ✅ Completed"
else
  echo "[FAST TEST] ❌ Failed with code $EXIT_CODE"
fi

