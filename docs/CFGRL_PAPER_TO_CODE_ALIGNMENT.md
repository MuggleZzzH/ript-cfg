# CFGRL 全面对齐改造方案（参考 cfgrl/ogbench/agents/cfgrl.py）

目标
- 将现有 PI0 + 序列 FM 训练/推理逻辑，全面对齐 OGBench 版 CFGRL 实现（`cfgrl/ogbench/agents/cfgrl.py`），做到“训练不加权 + 条件 dropout + 测试期 CFG 引导”的语义一致。
- 训练：单网络、FM（flow-matching）目标、对每个样本执行小概率条件 dropout，学习到“无条件(∅) + 条件(优化)”两个子分布。
- 推理：按 CFG（Classifier‑Free Guidance）合成速度场 `v = v_∅ + w·(v_1 − v_∅)`，仅通过测试期 `w` 控制策略改善幅度，无需再训练。

参考实现要点（cfgrl.py）
- 文件：`cfgrl/ogbench/agents/cfgrl.py:1`
- 训练（行为 FM）：
  - 采样 `x_0 ~ N(0, I)`、`t ~ U(0,1)`，构造 `x_t = (1−t)x_0 + t x_1`，目标速度 `vel = x_1 − x_0`；
  - 以概率 `p≈0.1` 执行条件 dropout：用“无条件嵌入”替换真实条件（目标/goal），得到 `goals = unc_embed or actor_goals`；
  - 单次前向：`pred = actor_flow(obs, x_t, t, goals)`，损失 `MSE(pred, vel)`；
  - 不做任何优势/权重加成，批内样本一视同仁。
- 推理（CFG 去噪）：
  - 初始化 `actions ~ N(0, I)`，迭代 `flow_steps` 次：
    - 计算 `unc_vel = v_θ(obs, a, t, o=∅)` 与 `cond_vel = v_θ(obs, a, t, o=1)`；
    - 合成 `v = unc_vel + w·(cond_vel − unc_vel)`，Euler 步进 `a ← a + v/flow_steps`；
  - `w` 为测试期超参（`cfg`），无需再训练即可扫提升幅度。

我们现状（关键文件）
- `ript-vla_ori/pi0/modeling_pi0.py:1`：PI0FlowMatching（PyTorch）前后缀组网、FM 目标、支持 `is_positive` 条件（token/bias/concat）。
- `ript-vla_ori/ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py:1`：当前“优势加权 + 双分支（pos/uncond=0）”的优化器实现。
- `ript-vla_ori/ript/algos/policies/pi0_oft_policy.py:1`：PI0 推理包装（目前未将 `cfg_scale` 透传给底层）。
- `ript-vla_ori/config/algo/pi0_cfg_rl.yaml:1`：算法配置（含 `alpha_uncond/use_binary_advantage` 等非 CFGRL 项）。

改造总览（论文 → 代码）
1) 去除优势加权，改为“无加权 + 条件 dropout”的单分支 FM 训练。
2) 训练样本仅携带 `o∈{1, ∅}`：正例代表“优化条件”，∅ 代表“无条件”；对 `o=1` 样本以概率 `p` 置为 `∅`（dropout）。
3) 推理严格采用 `v_∅/v_1` 的 CFG 合成；`w` 由推理配置或环境变量控制。
4) 语言/图像前缀维持现状（先实现“后缀无条件”即可）；如需更贴论文，可选做“前缀无条件 dropout”。

逐文件修改清单与建议

1) 模型（PI0FlowMatching）— `ript-vla_ori/pi0/modeling_pi0.py:500`
- 推理 CFG 合成（必须）：
  - 将“无条件分支”改为 `is_positive=None`（而非 `0`）；保持“条件分支”为 `is_positive=1`；
  - 计算 `v = v_∅ + cfg_scale · (v_1 − v_∅)`，Euler 步进；共享 prefix KV cache，仅替换 suffix 条件；
  - 目前实现已具备此路径，但 `v_uncond` 需确保走“不注入任何条件”的逻辑（`is_positive=None`）。
- 条件注入（必须）：
  - 在 `embed_suffix(...)` 中：当 `is_positive is None` 时，不注入任何条件（token/bias/concat 皆不走）；已默认落到“bias 无条件”分支，可保留；
  - 确认 token 模式下 `None` 不会构造 cond token（避免伪条件）。
- 可选增强（前缀无条件）：
  - 若要更贴论文 CF，可在 `prepare_language/embed_prefix` 中加入无条件分支（例如随机 mask 掉语言输入）；建议后续阶段再做。

2) RL 优化器（CFGRL 训练）— `ript-vla_ori/ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py:1`
- 训练范式（必须）：
  - 移除双分支两次前向与组合 `L = w_pos·L_pos + α·L_uncond`；
  - 单次前向：在 `_collate_samples(...)` 聚合时准备 `batch['is_positive']`：
    - 依据标签源生成 `o∈{1, ∅}`（见“标签生成”）；
    - 以 `dropout_prob` 将部分 `o=1` 样本置为 `∅`；
    - 前向 `model.model.forward(batch)`，其内部已按 `is_positive=None/1` 走对应条件；
  - 保留 AMP、梯度累积、DDP reduce、梯度裁剪等现有工程细节。
- 标签生成（建议）：
  - 选项 A（最小改动）：使用 episode 级 RLOO 优势 `A_ep` → 赋给窗口；`A_ep ≥ 0 ⇒ o=1` 否则 `∅`；
  - 选项 B（更细粒度）：使用窗口内奖励 `sum/mean` 或“窗口 RLOO”近似，得到 `A_win`；
  - 配置化：`cfgrl.label_source ∈ {episode_rloo, window_return, window_rloo}`、`cfgrl.label_threshold=0.0`。

3) 策略包装（透传推理超参）— `ript-vla_ori/ript/algos/policies/pi0_oft_policy.py:1`
- 推理函数 `select_action(...)`：
  - 将参数 `cfg_scale`、`is_positive_infer` 透传给底层 `self.model.select_action(...)`；
  - 支持从环境变量读取默认值（如 `PI0_CFG_SCALE`）。

4) 训练脚本与配置
- 配置文件 — `ript-vla_ori/config/algo/pi0_cfg_rl.yaml:1`
  - 新增 `cfgrl` 段：
    - `dropout_prob: 0.1`
    - `label_source: episode_rloo`  # 或 `window_return/window_rloo`
    - `label_threshold: 0.0`
  - 移除/忽略：`alpha_uncond`、`use_binary_advantage` 等优势加权相关字段；
  - 推理：保留/新增 `inference.cfg_scale`（默认 1.0，可在 rollout 时覆盖）。
- 训练脚本 — `ript-vla_ori/train_ript_pi0.py:1`
  - 无需大改；仍按批次调 `rl_optimizer.optimize(...)`；
  - 如果保留旧优化器，可新增一个 `cfgrl_optimizer_pi0.py` 并在 YAML 中切换 `_target_` 即可。

关键伪代码（映射论文 → 实现）
- 训练（窗口级 CFGRL）
```python
# 估计每个窗口标签 o
A = estimate_advantage(...)                   # episode_rloo / window_return / window_rloo
o = 1 if A >= label_threshold else None       # None 表示 ∅（无条件）
if o == 1 and rand() < dropout_prob:
    o = None

# 构造 batch 并单分支前向（FM）
batch = { 'image': ..., 'state': s0, 'action': a_seq, 'action_is_pad': mask,
          'prompt': [task], 'is_positive': (None if o is None else torch.ones(B, dtype=torch.long)) }
loss, _ = model.model.forward(batch)          # 内部：MSE(noise - action , v_t)
loss.backward(); optimizer.step()
```

- 推理（CFG 合成）
```python
v_uncond = predict_velocity(..., is_positive=None)
v_cond   = predict_velocity(..., is_positive=1)
v        = v_uncond + cfg_scale * (v_cond - v_uncond)
x_t     += dt * v
```

对齐检查清单（验收标准）
- 训练端：
  - 批内不再使用优势/α 权重；只存在 `o∈{1, ∅}` 与 `dropout_prob`；
  - `is_positive=None` 触发“真无条件”路径（不注入任何条件 token/偏置）；
  - FM 损失稳定下降，`o=∅` 占比接近设定的 dropout 概率。
- 推理端：
  - `cfg_scale` 从 1.0 增加时，成功率曲线在合理范围内上升后饱和/回落（存在最优 w）；
  - 仅在推理阶段调整 `w`，无需再训练即可改变策略偏好。

实现备注与风险
- 若暂不做“前缀无条件”，仅“后缀无条件”已可获得显著 CFG 效果；后续可再引入语言前缀 dropout 做精对齐。
- 使用 episode 级标签时，信号较粗；可在不稳定时尝试窗口级奖励或“窗口 RLOO”。
- `token`/`bias`/`concat` 三种条件注入方式语义等价但数值特性不同；建议默认 `token`，同时保留 `bias` 做消融。

快速使用建议
- 训练：
  - 在 YAML 中切换优化器为 CFGRL 版本（或为现有优化器增加 `mode: cfgrl` 开关），设定 `cfgrl.dropout_prob=0.1`；
  - 其他训练超参保持不变，先跑通稳定性。
- 推理：
  - 默认 `inference.cfg_scale=1.0`；离线扫 `w∈{1.0,1.5,2.0,3.0}` 观察最优点；
  - 如需从命令行临时覆盖，支持通过环境变量 `PI0_CFG_SCALE` 注入。

