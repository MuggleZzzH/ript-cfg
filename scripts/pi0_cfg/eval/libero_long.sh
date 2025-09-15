#!/bin/bash
set -eo pipefail

export LOG_BACKEND=${LOG_BACKEND:-swanlab}
export SWANLAB_MODE=${SWANLAB_MODE:-online}

export PI0_ENABLE_DUAL=${PI0_ENABLE_DUAL:-1}
export PI0_CFG_SCALE=${PI0_CFG_SCALE:-3.0}
export PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}
export PI0_N_VIDEO=${PI0_N_VIDEO:-1}

# Eval-time knobs
ROLLOUT_ENABLED=${ROLLOUT_ENABLED:-true}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-10}
NUM_ENVS=${NUM_ENVS:-1}
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-16}
MAX_EP_LEN=${MAX_EP_LEN:-300}
WAIT_STEPS=${WAIT_STEPS:-10}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
cd "$PROJECT_ROOT"

# Optional: conda
# 尝试多个可能的conda初始化路径
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
fi

# 设置默认环境为mix，可通过CONDA_ENV环境变量覆盖
CONDA_ENV=${CONDA_ENV:-mix}
[ -n "$CONDA_ENV" ] && conda activate "$CONDA_ENV" || true

if [ -z "$CKPT_DIR" ]; then
  echo "[!] Please set CKPT_DIR to a trained checkpoint directory (contains pi0_policy.pt)"
  exit 1
fi

MASTER_PORT=$(python - << 'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

echo "[*] Eval PI0 CFG (long, all tasks) | CKPT_DIR=$CKPT_DIR | CFG=$PI0_CFG_SCALE"
echo "[*] Eval params: num_envs=$NUM_ENVS, rollouts_per_env=$ROLLOUTS_PER_ENV, max_ep_len=$MAX_EP_LEN, wait=$WAIT_STEPS"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
python train_ript_pi0.py \
  --config-name=train_rl_pi0_cfg_all_task_long \
  algo.eval_only=true \
  rollout.enabled=$ROLLOUT_ENABLED \
  rollout.interval=$ROLLOUT_INTERVAL \
  algo.env_runner.num_parallel_envs=$NUM_ENVS \
  algo.rollouts_per_env=$ROLLOUTS_PER_ENV \
  algo.max_episode_length=$MAX_EP_LEN \
  algo.env_runner.num_steps_wait=$WAIT_STEPS \
  +rollout.n_video=$PI0_N_VIDEO \


echo "[*] Done."


