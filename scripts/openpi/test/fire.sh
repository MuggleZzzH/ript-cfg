#!/bin/bash

# ==============================================================================
# Bash 脚本: 训练 PI0 + CFG-Flow 模型 (LIBERO Spatial)
#
# 功能:
#   - 仿照 "perfect" 案例，实现健壮的脚本执行
#   - 双重输出: 使用 'tee' 命令将所有输出同时打印到终端并写入日志文件。
#   - 日志管理: 自动在 'training_logs/pi0_libero' 目录下创建带时间戳的日志文件。
#   - 健壮性设计: 任何命令失败时立即退出 (set -eo pipefail)。
#   - 动态路径: 自动定位项目根目录，确保脚本可在任何位置执行。
#   - 环境激活: 自动激活指定的 Conda 环境。
# ==============================================================================

# --- 脚本设置 ---
set -eo pipefail

# --- 步骤 1: 解析参数 & 设置日志文件 ---
usage() {
    cat <<EOF
用法: bash $(basename "$0") [选项]
  --steps N                 训练步数 (默认: 20)
  --rollout-enabled true|false  是否启用评测 (默认: true)
  --rollout-interval N      评测间隔步数 (默认: 10)
  --task NAME               任务配置 (默认: libero_spatial_rl)
  --cfg-scale S             推理CFG强度 (默认: 1.0)
  --is-positive 0|1         推理单分支条件 (仅token模式; 默认: 未设置)
  --condition-mode MODE     条件模式: bias|concat|token (默认: token)
  --pretrained PATH         预训练权重目录 (默认: 配置中的路径)
  --norm-stats PATH         归一化统计文件 (默认: 配置中的路径)
  --cuda IDS                CUDA_VISIBLE_DEVICES (默认: 保持当前)
  --wandb MODE              logging.mode: disabled|online|offline (默认: disabled)
  --accum-steps N           梯度累积步数 (默认: 1)
  --alpha-uncond A          无条件分支权重 alpha (默认: 0.1)
  --stride N                滑窗步长 stride (默认: 1)
  --max-windows N           每条episode最多窗口数 (默认: 不限)
  --opt-batch-size N        优化器微批大小 optimizer_batch_size (默认: 4)
  --batch-size N            训练 dataloader batch_size (默认: 使用配置)
  --rollouts-per-env N      每环境 rollout 数 (默认: 使用配置)
  --num-parallel-envs N     并行环境数 (默认: 使用配置)
  --max-episode-length N    每个 episode 最大步数 (默认: 使用配置)
  --eval-only true|false    仅评测，不训练 (默认: false)
  --config-name NAME        Hydra 顶层配置名 (默认: train_base_rl_openvla_oft)
  --offline true|false      Transformers 离线模式 (默认: false)
  --hf-home PATH            HF_HOME 缓存目录 (可选)
  -h, --help                显示帮助
EOF
}

# 默认参数
STEPS=${STEPS:-20}
ROLLOUT_ENABLED=${ROLLOUT_ENABLED:-true}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-10}
TASK=${TASK:-libero_spatial_rl}
PI0_CFG_SCALE=${PI0_CFG_SCALE:-1.0}
PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}
CONDITION_MODE=${CONDITION_MODE:-token}
PRETRAINED_PATH=${PRETRAINED_PATH:-}
NORM_STATS_PATH=${NORM_STATS_PATH:-}
WANDB_MODE=${WANDB_MODE:-disabled}
ACCUM_STEPS=${ACCUM_STEPS:-1}
ALPHA_UNCOND=${ALPHA_UNCOND:-0.1}
STRIDE=${STRIDE:-1}
MAX_WINDOWS=${MAX_WINDOWS:-}
OPT_BATCH_SIZE=${OPT_BATCH_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-}
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-}
NUM_PARALLEL_ENVS=${NUM_PARALLEL_ENVS:-}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTH:-}
EVAL_ONLY=${EVAL_ONLY:-false}
CONFIG_NAME=${CONFIG_NAME:-train_base_rl_openvla_oft}
OFFLINE=${OFFLINE:-false}
HF_HOME_PATH=${HF_HOME_PATH:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2;;
        --rollout-enabled) ROLLOUT_ENABLED="$2"; shift 2;;
        --rollout-interval) ROLLOUT_INTERVAL="$2"; shift 2;;
        --task) TASK="$2"; shift 2;;
        --cfg-scale) PI0_CFG_SCALE="$2"; shift 2;;
        --is-positive) PI0_IS_POSITIVE="$2"; shift 2;;
        --condition-mode) CONDITION_MODE="$2"; shift 2;;
        --pretrained) PRETRAINED_PATH="$2"; shift 2;;
        --norm-stats) NORM_STATS_PATH="$2"; shift 2;;
        --cuda) export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
        --wandb) WANDB_MODE="$2"; shift 2;;
        --accum-steps) ACCUM_STEPS="$2"; shift 2;;
        --alpha-uncond) ALPHA_UNCOND="$2"; shift 2;;
        --stride) STRIDE="$2"; shift 2;;
        --max-windows) MAX_WINDOWS="$2"; shift 2;;
        --opt-batch-size) OPT_BATCH_SIZE="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --rollouts-per-env) ROLLOUTS_PER_ENV="$2"; shift 2;;
        --num-parallel-envs) NUM_PARALLEL_ENVS="$2"; shift 2;;
        --max-episode-length) MAX_EPISODE_LENGTH="$2"; shift 2;;
        --eval-only) EVAL_ONLY="$2"; shift 2;;
        --config-name) CONFIG_NAME="$2"; shift 2;;
        --offline) OFFLINE="$2"; shift 2;;
        --hf-home) HF_HOME_PATH="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "未知参数: $1"; usage; exit 1;;
    esac
done

LOG_DIR="training_logs/pi0_libero"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_run_${TIMESTAMP}.log"

# --- 脚本主体通过 tee 同时输出到终端和日志文件 ---
{
    # --- 步骤 2: 定位项目根目录并切换 ---
    echo "--- 步骤 2: 定位项目根目录并切换工作目录 ---"
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
    PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
    cd "$PROJECT_ROOT"
    echo "✅ 已成功切换到项目目录: $(pwd)"
    echo "--- 所有日志将同时打印并在以下位置存档: $(pwd)/$LOG_FILE ---"
    echo

    # --- 步骤 3: 初始化并激活 Conda 环境 ---
    echo "--- 步骤 3: 初始化并激活 Conda 环境 ---"
    CONDA_INIT_SCRIPT="/opt/conda/etc/profile.d/conda.sh"
    if [ -f "$CONDA_INIT_SCRIPT" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_INIT_SCRIPT"
        echo "✅ Conda 初始化脚本加载成功。"
    else
        echo "❌ 错误: 未找到 Conda 初始化脚本 at '$CONDA_INIT_SCRIPT'。请检查路径。"
        exit 1
    fi
    conda activate mix
    echo "✅ Conda 环境 'mix' 已激活。"
    echo "   当前使用的 Python: $(which python)"
    echo

    # --- 步骤 4: 设置环境变量 ---
    echo "--- 步骤 4: 设置必要的环境变量 ---"
    export PI0_VERBOSE=1
    export HYDRA_FULL_ERROR=1
    export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT/LIBERO":"$PROJECT_ROOT"
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [[ "$OFFLINE" == "true" ]]; then
        export TRANSFORMERS_OFFLINE=1
        if [[ -n "$HF_HOME_PATH" ]]; then export HF_HOME="$HF_HOME_PATH"; fi
        echo "✅ Transformers 离线模式已启用 (TRANSFORMERS_OFFLINE=1)"
    fi
    # 推理CFG相关（runner中读取）
    export PI0_CFG_SCALE="$PI0_CFG_SCALE"
    if [[ -n "$PI0_IS_POSITIVE" ]]; then export PI0_IS_POSITIVE="$PI0_IS_POSITIVE"; fi
    # 可选：开启PI0推理过程的详细日志
    # export PI0_VERBOSE=1
    echo "✅ HYDRA_FULL_ERROR, PYTHONPATH, HF_ENDPOINT, PYTORCH_CUDA_ALLOC_CONF 已设置。"
    echo

    # --- 步骤 5: 准备并执行训练命令 ---
    echo "--- 步骤 5: 准备并执行 PI0 + CFG-Flow 训练命令 ---"
    MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    echo "   - 动态分配的主端口 (MASTER_PORT): $MASTER_PORT"
    echo "   - 使用本地模型权重和归一化统计..."
    echo "   - 训练配置: PI0 + CFG-Flow, ${TASK}, ${STEPS}步"
    echo "   - condition_mode=${CONDITION_MODE}, cfg_scale=${PI0_CFG_SCALE}, is_positive=${PI0_IS_POSITIVE:-none}"
    echo

    RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
    python train_ript_pi0.py \
      --config-name $CONFIG_NAME \
      algo=pi0_cfg_rl \
      $( [[ -n "$NORM_STATS_PATH" ]] && echo algo.norm_stats_path=$NORM_STATS_PATH || echo ) \
      $( [[ -n "$PRETRAINED_PATH" ]] && echo algo.policy.pretrained_path=$PRETRAINED_PATH || echo ) \
      training.n_steps=$STEPS \
      rollout.enabled=$ROLLOUT_ENABLED \
      rollout.interval=$ROLLOUT_INTERVAL \
      task=$TASK \
      algo.policy.condition_mode=$CONDITION_MODE \
      algo.gradient_accumulation_steps=$ACCUM_STEPS \
      algo.alpha_uncond=$ALPHA_UNCOND \
      algo.stride=$STRIDE \
      $( [[ -n "$MAX_WINDOWS" ]] && echo algo.max_windows_per_episode=$MAX_WINDOWS || echo ) \
      algo.rl_optimizer_factory.optimizer_batch_size=$OPT_BATCH_SIZE \
      $( [[ -n "$BATCH_SIZE" ]] && echo train_dataloader.batch_size=$BATCH_SIZE || echo ) \
      $( [[ -n "$ROLLOUTS_PER_ENV" ]] && echo algo.rollouts_per_env=$ROLLOUTS_PER_ENV || echo ) \
      $( [[ -n "$NUM_PARALLEL_ENVS" ]] && echo algo.num_parallel_envs=$NUM_PARALLEL_ENVS || echo ) \
      $( [[ -n "$MAX_EPISODE_LENGTH" ]] && echo algo.max_episode_length=$MAX_EPISODE_LENGTH || echo ) \
      algo.eval_only=$EVAL_ONLY \
      logging.mode=$WANDB_MODE

    echo
    echo "========================================================"
    echo "🎉 PI0 训练脚本执行完毕!"
    echo "--- 脚本结束: $(date) ---"
    echo "--- 完整的执行日志已保存在: $(pwd)/$LOG_FILE ---"
    echo "========================================================"

} 2>&1 | tee "$LOG_FILE"