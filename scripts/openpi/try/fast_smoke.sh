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

# --- CFG推理控制(环境变量) ---
# after conda activate mix
export LOG_BACKEND=swanlab
# 如果是自托管，还需要：
# export SWANLAB_HOST="http://your-host:port"
export PI0_ENABLE_DUAL=${PI0_ENABLE_DUAL:-1}    # 1=启用CFG双分支
export PI0_CFG_SCALE=${PI0_CFG_SCALE:-1.0}      # CFG引导权重(快速测试用较小值)
export PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}     # 空=正常CFG
export PI0_N_VIDEO=${PI0_N_VIDEO:-1}            # 录像数量
export PI0_VERBOSE=1
export DEBUG_SAVE_PROCESSED=1
export PI0_VIDEO_DIR=output/videos_pi0
mkdir -p "$PI0_VIDEO_DIR" || true
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT/LIBERO":"$PROJECT_ROOT"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PI0_DISABLE_INIT_STATES=0  # 快速测试：禁用初始状态设置
export MUJOCO_GL=egl

# --- Paths (edit if needed) ---
NORM_STATS=${NORM_STATS:-/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json}
PRETRAIN_PATH=${PRETRAIN_PATH:-/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch}

# --- Speed knobs (fast defaults) ---
TRAINING_STEPS=${TRAINING_STEPS:-2}             # total steps; cut=1 will stop after 1 loop
BATCH_SIZE=${BATCH_SIZE:-4}
RLOO_BATCH=${RLOO_BATCH:-4}
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-4}
NUM_ENVS=${NUM_ENVS:-4}
EARLY_STOP_PCT=${EARLY_STOP_PCT:-1}
ENABLE_DYNAMIC_SAMPLING=${ENABLE_DYNAMIC_SAMPLING:-false}
MAX_EP_LEN=${MAX_EP_LEN:-100}
WAIT_STEPS=${WAIT_STEPS:-10}
STRIDE=${STRIDE:-1}
MAX_WINDOWS=${MAX_WINDOWS:-}
OPTIMIZER_BATCH=${OPTIMIZER_BATCH:-20}
CF_DROPOUT_P=${CF_DROPOUT_P:-0.1}       # CF无分类器丢弃概率
CONDITION_MODE=${CONDITION_MODE:-token}
ROLLOUT_ENABLED=${ROLLOUT_ENABLED:-true}
EVAL_ONLY=${EVAL_ONLY:-false}
DDP_WRAP=${DDP_WRAP:-true}
NUM_GPUS=${NUM_GPUS:-3}

# CPU线程优化设置（根据你的CPU核心数调整）
# 建议设置为 CPU核心数 / GPU数量，避免线程竞争
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [ "$NUM_GPUS" -gt 1 ] && [ "$DDP_WRAP" != "true" ]; then
  echo "[FAST TEST] NUM_GPUS>1 detected; enabling DDP wrap"
  DDP_WRAP=true
fi

echo "[FAST TEST] Params: steps=$TRAINING_STEPS, batch=$BATCH_SIZE, rloo=$RLOO_BATCH, rollouts_per_env=$ROLLOUTS_PER_ENV"
echo "[FAST TEST] Episode: max_len=$MAX_EP_LEN, wait=$WAIT_STEPS; windows: stride=$STRIDE, max=$MAX_WINDOWS"
echo "[FAST TEST] CFG: dropout=$CF_DROPOUT_P, mode=$CONDITION_MODE"
echo "[FAST TEST] CFG推理: scale=$PI0_CFG_SCALE, is_positive=$PI0_IS_POSITIVE, dual=$PI0_ENABLE_DUAL"
echo "[FAST TEST] Dynamic sampling=$ENABLE_DYNAMIC_SAMPLING, early_stop=$EARLY_STOP_PCT, rollout.enabled=$ROLLOUT_ENABLED"
echo "[FAST TEST] Mode: eval_only=$EVAL_ONLY, ddp_wrap=$DDP_WRAP"
echo "[FAST TEST] GPUs: num_gpus=$NUM_GPUS"

# --- Distributed single-process wiring ---
MASTER_PORT=$(python - << 'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

COMMON_ARGS=(
  --config-name=train_base_rl_pi0_cfg
  algo.norm_stats_path=$NORM_STATS
  algo.policy.pretrained_path=$PRETRAIN_PATH
  training.n_steps=$TRAINING_STEPS
  train_dataloader.batch_size=$BATCH_SIZE
  algo.rloo_batch_size=$RLOO_BATCH
  algo.rollouts_per_env=$ROLLOUTS_PER_ENV
  algo.env_runner.num_parallel_envs=$NUM_ENVS
  algo.early_stop_percentage=$EARLY_STOP_PCT
  algo.enable_dynamic_sampling=$ENABLE_DYNAMIC_SAMPLING
  algo.max_episode_length=$MAX_EP_LEN
  algo.env_runner.num_steps_wait=$WAIT_STEPS
  algo.stride=$STRIDE
  algo.max_windows_per_episode=$MAX_WINDOWS
  algo.optimizer_batch_size=$OPTIMIZER_BATCH
  algo.cf_dropout_p=$CF_DROPOUT_P
  algo.policy.condition_mode=$CONDITION_MODE
  algo.eval_only=$EVAL_ONLY
  +algo.policy.ddp_wrap=$DDP_WRAP
  rollout.enabled=$ROLLOUT_ENABLED
  +logging.backend=swanlab
  +rollout.n_video=1
  'task.task_names_to_use=["pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate","pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate","pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate","pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]'
)

if [ "$NUM_GPUS" -gt 1 ]; then
  echo "[FAST TEST] Launching torchrun with $NUM_GPUS GPUs (master_port=$MASTER_PORT)"
  torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    train_ript_pi0.py "${COMMON_ARGS[@]}"
  EXIT_CODE=$?
else
  echo "[FAST TEST] Launching python with single GPU (master_port=$MASTER_PORT)"
  MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT RANK=0 WORLD_SIZE=1 python train_ript_pi0.py "${COMMON_ARGS[@]}"
  EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
  echo "[FAST TEST] ✅ Completed"
else
  echo "[FAST TEST] ❌ Failed with code $EXIT_CODE"
fi

