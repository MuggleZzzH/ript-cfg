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

# --- 步骤 1: 设置日志文件 ---
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

    # --- 步骤 4: 设置环境变量和训练参数 ---
    echo "--- 步骤 4: 设置必要的环境变量和训练参数 ---"
    export PI0_VERBOSE=1
    export HYDRA_FULL_ERROR=1
    export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT/LIBERO":"$PROJECT_ROOT"
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # === 日志后端设置 ===
    export LOG_BACKEND=swanlab # <--- 设置日志后端为 swanlab
    # export SWANLAB_MODE=offline # <--- 如果需要离线模式，取消此行注释

    # === 关键训练参数配置 ===
    TRAINING_STEPS=2                    # 训练步数
    BATCH_SIZE=2                         # 批次大小  
    GRADIENT_ACCUM_STEPS=4               # 梯度累积步数
    LEARNING_RATE=2.5e-5                 # 学习率
    
    # === CFG-Flow 训练参数 ===
    CONDITION_MODE=${CONDITION_MODE:-"token"}        # CFG注入模式: bias|token|concat
    CF_DROPOUT_P=${CF_DROPOUT_P:-0.1}               # CF无分类器丢弃概率
    STRIDE=${STRIDE:-1}                             # 滑窗步长
    MAX_WINDOWS=${MAX_WINDOWS:-}                    # 每episode最大窗口数(空=自适应)
    OPTIMIZER_BATCH_SIZE=${OPTIMIZER_BATCH_SIZE:-4} # 优化器微批大小
    
    # === CFG推理控制(环境变量,runner会读取) ===
    export PI0_ENABLE_DUAL=${PI0_ENABLE_DUAL:-1}    # 1=启用CFG双分支, 0=单分支兼容模式
    export PI0_CFG_SCALE=${PI0_CFG_SCALE:-3.0}      # CFG引导权重w(对应论文guidance weight)
    export PI0_IS_POSITIVE=${PI0_IS_POSITIVE:-}     # 强制分支:空=正常CFG,1=正分支,0=负分支
    export PI0_N_VIDEO=${PI0_N_VIDEO:-0}            # 评测录像数量
    
    # === 评测参数 ===
    ROLLOUT_ENABLED=true                 # 是否启用评测
    ROLLOUT_INTERVAL=2                  # 评测间隔
    
    # === 环境与并行设置 ===
    NUM_PARALLEL_ENVS=2                  # 并行环境数

    ROLLOUTS_PER_ENV=16                  # 每环境rollout数
    MAX_EPISODE_LENGTH=300               # 最大episode长度
    
    echo "✅ 训练参数配置完成:"
    echo "   - 训练步数: $TRAINING_STEPS, 批次大小: $BATCH_SIZE"
    echo "   - CFG训练: mode=$CONDITION_MODE, dropout=$CF_DROPOUT_P"
    echo "   - CFG推理: scale=$PI0_CFG_SCALE, is_positive=$PI0_IS_POSITIVE, dual=$PI0_ENABLE_DUAL"
    echo "   - 评测: enabled=$ROLLOUT_ENABLED, interval=$ROLLOUT_INTERVAL, video=$PI0_N_VIDEO"
    echo

    # --- 步骤 5: 准备并执行训练命令 ---
    echo "--- 步骤 5: 准备并执行 PI0 + CFG-Flow 训练命令 ---"
    MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    echo "   - 动态分配的主端口 (MASTER_PORT): $MASTER_PORT"
    echo "   - 使用本地模型权重和归一化统计..."
    echo "   - 训练配置: PI0 + CFG-Flow, libero_spatial_rl, ${TRAINING_STEPS}步"
    echo
    
    RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
    python train_ript_pi0.py \
      --config-name=train_base_rl_openvla_oft \
      algo=pi0_cfg_rl \
      algo.norm_stats_path=/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json \
      algo.policy.pretrained_path=/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch \
      training.n_steps=$TRAINING_STEPS \
      train_dataloader.batch_size=$BATCH_SIZE \
      algo.gradient_accumulation_steps=$GRADIENT_ACCUM_STEPS \
      algo.lr=$LEARNING_RATE \
      algo.policy.condition_mode=$CONDITION_MODE \
      algo.cf_dropout_p=$CF_DROPOUT_P \
      algo.stride=$STRIDE \
      $( [ -n "$MAX_WINDOWS" ] && echo "algo.max_windows_per_episode=$MAX_WINDOWS" ) \
      algo.optimizer_batch_size=$OPTIMIZER_BATCH_SIZE \
      rollout.enabled=$ROLLOUT_ENABLED \
      rollout.interval=$ROLLOUT_INTERVAL \
      algo.num_parallel_envs=$NUM_PARALLEL_ENVS \
      algo.rollouts_per_env=$ROLLOUTS_PER_ENV \
      algo.max_episode_length=$MAX_EPISODE_LENGTH \
      task=libero_spatial_rl \
      # logging.mode=disabled # <--- 已删除此行以启用日志记录

    echo
    echo "========================================================"
    echo "🎉 PI0 训练脚本执行完毕!"
    echo "--- 脚本结束: $(date) ---"
    echo "--- 完整的执行日志已保存在: $(pwd)/$LOG_FILE ---"
    echo "========================================================"

} 2>&1 | tee "$LOG_FILE"