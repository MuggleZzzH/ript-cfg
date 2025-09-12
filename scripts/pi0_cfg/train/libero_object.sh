#!/bin/bash
set -eo pipefail

# Env backend: swanlab|wandb|none (default wandb)
export LOG_BACKEND=${LOG_BACKEND:-swanlab}
export SWANLAB_MODE=${SWANLAB_MODE:-online}

# CFG runtime controls
export PI0_ENABLE_DUAL=${PI0_ENABLE_DUAL:-1}
export PI0_CFG_SCALE=${PI0_CFG_SCALE:-1.0}
export PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}

# Video off for training; set to 1 to enable
export PI0_N_VIDEO=${PI0_N_VIDEO:-0}

# Train-time knobs (override via env before calling this script)
TRAINING_STEPS=${TRAINING_STEPS:-12}
BATCH_SIZE=${BATCH_SIZE:-24}
GRADIENT_ACCUM_STEPS=${GRADIENT_ACCUM_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-2.5e-5}
RLOO_BATCH=${RLOO_BATCH:-8}
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-50}
NUM_ENVS=${NUM_ENVS:-5}
EARLY_STOP_PCT=${EARLY_STOP_PCT:-1.0}
ENABLE_DYNAMIC_SAMPLING=${ENABLE_DYNAMIC_SAMPLING:-true}
MAX_EP_LEN=${MAX_EP_LEN:-null}
WAIT_STEPS=${WAIT_STEPS:-10}
STRIDE=${STRIDE:-1}
MAX_WINDOWS=${MAX_WINDOWS:-}
OPTIMIZER_BATCH=${OPTIMIZER_BATCH:-20}
CF_DROPOUT_P=${CF_DROPOUT_P:-0.1}
CONDITION_MODE=${CONDITION_MODE:-token}
ROLLOUT_ENABLED=${ROLLOUT_ENABLED:-true}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-4}
NORM_STATS=${NORM_STATS:-"/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json"}
PRETRAIN_PATH=${PRETRAIN_PATH:-"/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_base_pytorch"}

# 动态生成实验名称
EXP_NAME=${EXP_NAME:-"pi0_cfg_object_cfg${PI0_CFG_SCALE}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_drop${CF_DROPOUT_P}"}
VARIANT_NAME=${VARIANT_NAME:-"steps${TRAINING_STEPS}_envs${NUM_ENVS}_rollout${ROLLOUTS_PER_ENV}"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
cd "$PROJECT_ROOT"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
fi
[ -n "$CONDA_ENV" ] && conda activate "$CONDA_ENV" || true

MASTER_PORT=$(python - << 'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

echo "[*] Start PI0 CFG training (object, all tasks) | LOG_BACKEND=$LOG_BACKEND | CFG=$PI0_CFG_SCALE"
echo "[*] Experiment: $EXP_NAME | Variant: $VARIANT_NAME"
echo "[*] Train params: steps=$TRAINING_STEPS, batch=$BATCH_SIZE, lr=$LEARNING_RATE, accum=$GRADIENT_ACCUM_STEPS"
echo "[*] RL: rloo=$RLOO_BATCH, rollouts_per_env=$ROLLOUTS_PER_ENV, num_envs=$NUM_ENVS, max_ep_len=$MAX_EP_LEN, wait=$WAIT_STEPS"
echo "[*] Windows: stride=$STRIDE, max_windows=${MAX_WINDOWS:-auto}, opt_batch=$OPTIMIZER_BATCH, cf_dropout=$CF_DROPOUT_P, mode=$CONDITION_MODE"
echo "[*] Rollout: enabled=$ROLLOUT_ENABLED, interval=$ROLLOUT_INTERVAL, videos=$PI0_N_VIDEO"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
python train_ript_pi0.py \
  --config-name=train_rl_pi0_cfg_all_task_object \
  exp_name="$EXP_NAME" \
  variant_name="$VARIANT_NAME" \
  $( [ -n "$NORM_STATS" ] && echo "algo.norm_stats_path=$NORM_STATS" ) \
  $( [ -n "$PRETRAIN_PATH" ] && echo "algo.policy.pretrained_path=$PRETRAIN_PATH" ) \
  training.n_steps=$TRAINING_STEPS \
  train_dataloader.batch_size=$BATCH_SIZE \
  algo.gradient_accumulation_steps=$GRADIENT_ACCUM_STEPS \
  algo.lr=$LEARNING_RATE \
  algo.rloo_batch_size=$RLOO_BATCH \
  algo.rollouts_per_env=$ROLLOUTS_PER_ENV \
  algo.env_runner.num_parallel_envs=$NUM_ENVS \
  algo.early_stop_percentage=$EARLY_STOP_PCT \
  algo.enable_dynamic_sampling=$ENABLE_DYNAMIC_SAMPLING \
  algo.max_episode_length=$MAX_EP_LEN \
  algo.env_runner.num_steps_wait=$WAIT_STEPS \
  algo.stride=$STRIDE \
  $( [ -n "$MAX_WINDOWS" ] && echo "algo.max_windows_per_episode=$MAX_WINDOWS" ) \
  algo.optimizer_batch_size=$OPTIMIZER_BATCH \
  algo.cf_dropout_p=$CF_DROPOUT_P \
  algo.policy.condition_mode=$CONDITION_MODE \
  algo.eval_only=false \
  rollout.enabled=$ROLLOUT_ENABLED \
  rollout.interval=$ROLLOUT_INTERVAL \
  +rollout.n_video=$PI0_N_VIDEO \


echo "[*] Done."


