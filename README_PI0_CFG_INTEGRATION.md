## 在 RIPT 中集成 PI0 与基于优势权重的 CFG-Flow 训练（修订版实施说明）

本说明基于原生 RIPT、PI0（`openpi_pytorch_ori`）与 CFG（参考 `cfgrl/rlbase/algs_offline/iql_diffusion.py`）重新校准整体方案，明确采用“多步（50 步）序列执行 + 优势加权的 CFG-Flow 优化”，并尽可能复用 RIPT 的并发与评测基础设施。

### 关键修订点（与旧稿差异）
- 必须执行并监督完整 50 步动作序列（`n_action_steps=50`），而非单步 open-loop=1。
- 优势权重仅作用于“条件（is_positive=1）”分支损失；无条件分支使用固定小权重（如 0.1），不乘优势。
- 训练样本来自 episode 的时间切片：每 50 步切出一条监督样本，继承该 episode 的优势权重。

---

## 一、组件复用与替换

### 可直接复用（无改动）
- `ript/algos/rl_optimizers/rollout_generator.py`
- `ript/algos/rl_optimizers/file_counter.py`
- 训练主流程与分布式/日志：`train_ript_openvla_oft.py`

### 需要新增/替换（最小集合）
1) 策略包装器：`ript/algos/policies/pi0_oft_policy.py`
- 封装 `openpi_pytorch_ori/pi0/PI0Policy`，提供：
  - `self.model`（可 DDP 包装）
  - `self.trainable_params['model']`（供优化器分组）
  - `select_action(observation, cfg_scale)`：返回 `(B, 50, 7)` 序列
  - 维护 `norm_stats`（state/action mean/std）

2) 观测适配：`ript/env_runner/pi0_runner_utils.py`
- 从环境观测构造 PI0 期望的 observation：
  - 图像键映射：`agentview_image` → `image.base_0_rgb`，`robot0_eye_in_hand_image` → `image.left_wrist_0_rgb`
  - 状态：`eef_pos + axisangle(eef_quat) + gripper`（8 维），用 `norm_stats` 归一化
  - 语言：`prompt=[task_description]`
- 图像大小与归一化交由 `PI0Policy.prepare_images` 处理（runner 不做 resize/pad）

3) Runner：`ript/env_runner/pi0_libero_runner.py`
- 去除 `experiments.robot.*` 与 `prismatic.vla.*` 依赖，改用 `pi0_runner_utils`
- 执行策略：每 50 步调用一次 `policy.select_action(...)`，随后按序执行 50 步至环境（open-loop=50）
- Episode 字段：至少包含 `actions`、`observations`、`valid`、`task_description`

4) 新优化器：`ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py`
- 用 RLOO 计算优势；基于优势权重训练 PI0 的 Flow Matching：
  - 条件分支（is_positive=1）：损失乘以优势权重
  - 无条件分支（is_positive=0）：损失乘以固定系数（如 0.1）
- 接口对齐：`optimize(model, batch, optimizers, data_iterator, dataloader)`

---

## 二、目录与配置改动

新增文件（建议路径）：
- `ript/algos/policies/pi0_oft_policy.py`
- `ript/env_runner/pi0_runner_utils.py`
- `ript/env_runner/pi0_libero_runner.py`
- `ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py`

Hydra 配置（示意）：
```
algo:
  policy:
    _target_: ript.algos.policies.pi0_oft_policy.PI0_OFT_Policy
    norm_stats_path: <REPLACE_WITH_norm_stats.json>
  env_runner:
    _target_: ript.env_runner.pi0_libero_runner.Pi0LiberoRunner
    num_parallel_envs: 1
    max_episode_length: 300
  rollout_generator_factory:
    _target_: ript.algos.rl_optimizers.RolloutGenerator
  rl_optimizer_factory:
    _target_: ript.algos.rl_optimizers.cfg_flow_optimizer_pi0.CFGFlowOptimizerPI0
training:
  n_steps: 100000
```

训练入口：继续使用 `train_ript_openvla_oft.py`（装配与分布式/日志逻辑保持不变）。

---

## 三、实现要点与接口约束

### 1) PI0_OFT_Policy
- `__init__(device_id, norm_stats_path, ...)`：加载 `PI0Policy` 与 `norm_stats.json`；构造 `trainable_params['model']`
- `select_action(observation, cfg_scale)`：返回 `(B, 50, 7)`；推理输出为归一化动作，runner 需在执行前反归一化

### 2) Pi0LiberoRunner（open-loop=50）
- 每当 `t % 50 == 0` 时：
  - 用 `pi0_runner_utils` 将当前观测构造成 PI0 observation
  - `action_seq = policy.select_action(obs, cfg_scale)`，得到 50 步动作序列
- 对接下来的 50 个环境步：
  - 取 `action = action_seq[local_step]`
  - 使用 `norm_stats` 反归一化后执行到环境
  - 记录 `actions/observations/valid` 等
- 注意：若 episode 提前结束（done），应跳出循环

### 3) Episode → 训练样本切片（对齐 50 步）
- 每个 episode（可能 200+ 步）需切分为多条训练样本：`[t : t+50)`，步长 50
- 每条样本包含：
  - 起始观测（构造成 PI0 observation）
  - 50 步动作序列（归一化后作为监督标签）
  - 继承该 episode 的优势权重（见下）

### 4) CFGFlowOptimizerPI0（优势加权位置与分支）
- 优势计算：采用 RLOO（leave-one-out）
  - `advantage = r_i - mean_{j≠i}(r_j)`（按 demo_batch × rloo_batch 维度重排）
- 样本构造：按“3) 切片”规则生成；动作先用 `norm_stats` 归一化
- 损失组合：
  - 若 `advantage > 0`：计算条件分支损失 `L_pos`（is_positive=1），并用优势权重加权（可截断/温度缩放）
  - 始终计算无条件分支损失 `L_uncond`（is_positive=0），乘固定系数 `α≈0.1`
  - `L = w(adv) * L_pos + α * L_uncond`
- 反传与分布式规约：参考 `RLOptimizerOpenVLAOFT` 的 `gradient_accumulation + dist.all_reduce + clip + step`

---

## 四、规范与默认值
- PI0 配置：`n_action_steps=50`；`resize_imgs_with_padding=(224,224)`
- Runner：`open_loop=50`；`num_parallel_envs` 依资源设置
- 优势权重：建议归一化/截断至 `[0, a_max]`（如 `a_max=5`）
- 无条件权重：`α=0.1`
- CFG 推理：训练阶段 `cfg_scale=1.0`，评测可调 `[1.0, 3.0]`
- 归一化：确保使用与 PI0 预训练一致的 `norm_stats.json`（如 `.../libero/norm_stats.json`）

---

## 五、验证与测试
1) 单任务/单 GPU 冒烟：确认 50 步序列生成与执行无误（无键名/shape 报错）
2) 100–500 步训练观测：`mean_advantage/mean_scores/fm_loss` 是否合理
3) 评测：开启 `cfg.rollout.enabled`，观察 SR/回报提升

---

## 六、风险与对策
- 序列对齐：必须严格以 50 步为窗口切片，避免“预测 50 步、执行 1 步”的分布偏移
- 归一化一致性：动作/状态的归一化与反归一化必须与 `norm_stats` 对齐
- 提前终止：episode 短于 50 步时，需跳过或补齐（建议跳过，避免伪标签）
- 优势数值稳定：建议做标准化/截断，避免梯度爆

---

## 七、落地步骤（执行清单）
1) 新增 4 个文件：`pi0_oft_policy.py`、`pi0_runner_utils.py`、`pi0_libero_runner.py`、`cfg_flow_optimizer_pi0.py`
2) 更新 Hydra 配置：将 policy/runner/optimizer 指向新组件；设置 `norm_stats_path`
3) 跑通 `train_ript_openvla_oft.py`（单任务→多任务）
4) 调参：`cfg_scale`、优势截断阈值 `a_max`、无条件权重 `α`

---

## 八、FAQ
- 是否需要改 `train_ript.py`？不需要。该脚本为 token/离散动作线；PI0 走 `train_ript_openvla_oft.py` 连续动作线。
- 是否可以直接沿用 OpenVLA 的工具？不建议。改用 `pi0_runner_utils`，避免额外依赖。



---

## 九、关键缺失补全（必须实现，确保方案可落地）

1) 模型侧：为 PI0 增加 CFG 的条件嵌入（is_positive）
- 目的：对齐 `iql_diffusion.py` 的“条件/无条件”双分支，与共享 noise/time 的 CFG 组合。
- 最小改动（建议在 `PI0FlowMatching` 内实现）：
  - 定义 `self.cond_embed = nn.Embedding(2, self.config.proj_width)`；
  - 在 `embed_suffix(state, noisy_actions, timestep, is_positive)` 中：
    - `pos_emb = self.cond_embed(is_positive.long()).unsqueeze(1).expand(-1, noisy_actions.shape[1], -1)`；
    - 将 `pos_emb` 与 `action_time_emb` 融合（相加或 concat 后过一层线性变换），最终再走 `action_time_mlp_in/out`；
  - `forward(...)` 与 `sample_actions(...)` 调用 `embed_suffix` 时，传入 `is_positive`（训练阶段：条件/无条件各一次；推理保持条件分支）。
- 关键细节：条件/无条件两次前向必须共享同一 `noise` 与 `time`，完全对齐 `iql_diffusion.py`。

2) 训练脚本兼容性：提供 `trainable_params` 的两组参数
- `train_ript_openvla_oft.py` 固定创建两组优化器并分别做 grad clip 与 step：
  - `model.trainable_params['model']`
  - `model.trainable_params['header']`
- PI0 的策略包装器需按此约定暴露参数分组（不改训练脚本）：
  - 建议：`model` 组包含 PI0 主干与大多数可训练层；`header` 组包含较小的头部/投影层（如 `action_out_proj`、`state_proj`），或单独的轻量 MLP；
  - 避免空参数组（空列表会导致优化器报错）。

3) 归一化统计：显式可配的 `norm_stats.json`
- 必须从 PI0 训练资产中加载与 LIBERO 对齐的统计（state[:8], action[:7]）。
- 在策略包装器中提供 `norm_stats_path`（Hydra 配置传入），若未显式配置，则按常见路径回退查找；
- Runner 执行前反归一化动作（与 `openpi_pytorch/2_test_pi0_on_libero.py` 一致）。


---

## 十、Hydra 配置改动（可直接复制使用）

建议新增 `config/algo/pi0_cfg_rl.yaml`，并在 `train_base_rl_openvla_oft.yaml` 中将 `algo: openvla_oft_rl` 改为 `algo: pi0_cfg_rl`。

```yaml
defaults:
  - _self_

policy:
  _target_: ript.algos.policies.pi0_oft_policy.PI0_OFT_Policy
  device_id: 0
  norm_stats_path: /ABS/PATH/TO/norm_stats.json   # 必填：与PI0预训练资产一致

optimizer_factory:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 2.5e-5
  betas: [0.9, 0.95]
  weight_decay: 1.0e-10

scheduler_factory:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  _partial_: true
  eta_min: 2.5e-6
  last_epoch: -1
  T_max: ${training.n_epochs}

env_runner:
  _target_: ript.env_runner.pi0_libero_runner.Pi0LiberoRunner
  num_parallel_envs: ${algo.num_parallel_envs}
  max_episode_length: ${algo.max_episode_length}

rollout_generator_factory:
  _target_: ript.algos.rl_optimizers.rollout_generator.RolloutGenerator
  _partial_: true
  rloo_batch_size: ${algo.rloo_batch_size}
  demo_batch_size: ${train_dataloader.batch_size}
  use_tqdm: true
  early_stop_percentage: ${algo.early_stop_percentage}
  enable_dynamic_sampling: ${algo.enable_dynamic_sampling}
  use_val_init: ${algo.use_val_init}
  mix_val_init_in_rloo: ${algo.mix_val_init_in_rloo}

rl_optimizer_factory:
  _target_: ript.algos.rl_optimizers.cfg_flow_optimizer_pi0.CFGFlowOptimizerPI0
  _partial_: true
  gradient_accumulation_steps: ${algo.gradient_accumulation_steps}
  # 其余超参（如 a_max, cfg_uncond_weight 等）在该优化器内部读取

name: pi0_cfg

# 训练/评估超参
num_parallel_envs: 1
rollouts_per_env: 16
max_episode_length: 300
rloo_batch_size: 8
early_stop_percentage: 1.0
enable_dynamic_sampling: false
use_val_init: false
mix_val_init_in_rloo: false

gradient_accumulation_steps: 1
eval_only: false

model_seed: 7
fix_scale_head: false
```

说明：
- 将 `train_ript_openvla_oft.py` 的主流程与分布式/日志保持不变，仅通过该配置切换 policy/runner/optimizer。
- 策略包装器需要提供 `trainable_params['model']` 与 `['header']` 两组参数以兼容训练脚本。


---

## 十一、替换点总表（对照原 OpenVLA-OFT 版本）

- policy：
  - 从 `ript.algos.rl_optimizers.openvla_oft_interface.OpenVLA_OFT_Policy`
  - 替换为 `ript.algos.policies.pi0_oft_policy.PI0_OFT_Policy`
- env_runner：
  - 从 `ript.env_runner.openvla_oft_libero_runner.OpenVLAOFTLiberoRunner`
  - 替换为 `ript.env_runner.pi0_libero_runner.Pi0LiberoRunner`（open-loop=50、按窗口切片样本）
- rl_optimizer：
  - 从 `ript.algos.rl_optimizers.rl_optimizer_openvla_oft.RLOptimizerOpenVLAOFT`（PPO）
  - 替换为 `ript.algos.rl_optimizers.cfg_flow_optimizer_pi0.CFGFlowOptimizerPI0`（优势加权 CFG-Flow）
- 关键行为变化：
  - 执行序列改为 50 步；
  - 优势权重仅作用于条件分支；无条件分支固定权重 α≈0.1；
  - 训练样本按 50 步窗口从 episode 切片；
  - 条件/无条件分支共享 noise/time（严格对齐 `iql_diffusion.py`）。


---

## 十二、最小验证用例（建议先本地单卡跑通）

---

## 十三、数据采样与监督细则（对齐 openpi）

- 滑窗采样（stride=1）：对单条轨迹长度 N、动作视界 H=50，样本起点 t 取 [0, 1, …, N−H]，每个样本含 obs[t] 与 actions[t:t+H]。为控内存可用 stride=10/50 或 last 模式。
- 相对动作监督：label 动作为“相对量”，由绝对动作与起点状态还原 delta；再按 `norm_stats` 做归一化。执行时先反归一化，再将前 6 维加回起点 eef 位姿，得到绝对动作。
- padding 掩码：短于 H 的窗口以 `action_is_pad` 标记并在损失归约时屏蔽。
- 语言/图像：图像 HWC uint8，策略内部做 normalize/resize/pad；prompt 直接传递字符串列表。

1) 数据/环境：单任务、单 GPU，`num_parallel_envs=1`，`rollouts_per_env=4`；
2) 训练 50〜200 步，检查：
   - 日志中 `mean_advantage`、`mean_scores`、`fm_loss` 是否合理收敛；
   - runner 是否严格按 50 步序列执行，短 episode 是否跳过（不做 padding 伪标签）；
3) 评测：`cfg.rollout.enabled=true`，调整 `cfg_scale ∈ [1.0, 3.0]` 观察成功率提升趋势。

