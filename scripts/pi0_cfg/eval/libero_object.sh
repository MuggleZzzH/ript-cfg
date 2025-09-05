#!/bin/bash
set -eo pipefail

export LOG_BACKEND=${LOG_BACKEND:-swanlab}
export SWANLAB_MODE=${SWANLAB_MODE:-online}

export PI0_ENABLE_DUAL=${PI0_ENABLE_DUAL:-1}
export PI0_CFG_SCALE=${PI0_CFG_SCALE:-3.0}
export PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}
export PI0_N_VIDEO=${PI0_N_VIDEO:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
cd "$PROJECT_ROOT"

if [ -z "$CKPT_DIR" ]; then
  echo "[!] Please set CKPT_DIR to a trained checkpoint directory (contains pi0_policy.pt)"
  exit 1
fi

MASTER_PORT=$(python - << 'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

echo "[*] Eval PI0 CFG (object, all tasks) | CKPT_DIR=$CKPT_DIR | CFG=$PI0_CFG_SCALE"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
python train_ript_pi0.py \
  --config-name=train_rl_pi0_cfg_all_task_object \
  algo.eval_only=true \
  rollout.enabled=true \
  +rollout.n_video=$PI0_N_VIDEO \
  logging.mode=disabled

echo "[*] Done."


