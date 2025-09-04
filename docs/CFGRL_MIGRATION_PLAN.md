# CFGRL 全量对齐改造方案（PI0 + 序列FM）

目标
- 将当前基于“优势门控/加权的双分支FM”实现，迁移为与 cfgrl.py 思想一致的“Classifier‑Free Guidance RL（CFGRL）”范式。
- 训练：无加权的条件/无条件监督（含条件 dropout）；推理：cond/∅ 双分支线性引导；动作从“单步”扩展为“长度 H 的动作序列”保持不变。

现状 vs 目标（高层对比）
- 现状训练
  - 用 episode 级 RLOO 优势，构造 `w_pos`（二值或连续）门控，组合 `L = w_pos·L_pos + α·L_uncond`。
  - “无条件”分支使用 `is_positive=0`（后缀负类），并未做严格 CF 的“无条件（∅）”。
- 目标训练（CFGRL）
  - 不对损失加权；为每个样本生成条件标签 `o∈{1, ∅}`（可选扩展到 `{0,1,∅}`）。
  - 以概率 `p` 做条件 dropout：当样本是 `o=1` 时，以 `p` 概率将其置为 `o=∅` 用于训练无条件先验。
  - 单分支前向：`v_θ(a_t, t, s, o)`，最小二乘流匹配（FM）损失。
- 现状推理
  - `v = v_uncond(is_positive=0) + w·(v_pos(is_positive=1) − v_uncond)`（仅后缀条件切换）。
- 目标推理（CFGRL）
  - `v = v_∅ + w·(v_1 − v_∅)`，其中 `v_∅` 是“真正无条件（∅）”：不注入任何条件 token/偏置（至少在后缀；可选前缀也 drop）。

需要修改的模块与要点

1) 模型与策略（`ript-vla_ori/pi0/modeling_pi0.py`）
- 无条件语义（必须）
  - 在 `embed_suffix(...)` 中：当 `o=∅`（可用 `is_positive=None` 表示）时，严格“不注入任何条件”。
  - 当 `o=1` 时，正常注入条件（token/bias/concat 任一）。
- 推理期双分支（必须）
  - `sample_actions(...)` 中：
    - 计算 `v_∅ = predict_velocity(..., is_positive=None)`。
    - 计算 `v_1 = predict_velocity(..., is_positive=1)`。
    - 融合：`v = v_∅ + w·(v_1 − v_∅)`，`w=cfg_scale`。
  - 共享 prefix KV cache；仅切换 suffix 条件（与现有结构一致）。
- 可选：前缀无条件（建议但可延后）
  - 若要更贴论文 CF：在无条件时对前缀（语言）也做 dropout（不注入 prompt token 或缩短到最小），需要在 `prepare_language` / prefix embed 路径增加 o=∅ 分支。

2) 环境与样本生成（`ript-vla_ori/ript/env_runner/pi0_libero_runner.py` + rollout generator）
- 记录逐步奖励/终止（建议）
  - 在 `env.step` 循环中把 `reward/done` 逐步追加到 `episode['rewards'] / ['dones']`，用于后续窗口/步级标签计算。
- 窗口化样本（保留）
  - 继续使用 H=50 的动作序列窗口；保存 `action_is_pad` 掩码、起始观测 `obs0`、标准化相对动作等。
- 标签来源（CFGRL）
  - 推荐“窗口级”标签：对每个窗口估计一个优势标量（如下 A_w）→ `A_w ≥ 0 ⇒ o=1；否则 o=∅`；
  - 估计方式（多选其一）：
    - 使用 episode 级 RLOO 优势值填充到窗口（现有最小改动）。
    - 使用窗口内奖励之和/平均、或“窗口内部 RLOO”近似。
  - 条件 dropout：以概率 `p` 将 `o=1` 样本替换为 `o=∅`，比例建议 `p≈0.1`。

3) 优化器（`ript-vla_ori/ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py`）
- 去除优势加权（必须）
  - 移除 `w_pos`、`alpha_uncond` 组合；不再两次前向（pos/uncond）。
- 单分支前向（必须）
  - 对每个样本：根据标签 `o∈{1, ∅}` 设定 `is_positive=None/1`，调用一次 `model.model.forward(batch)`（FM MSE）。
  - 继续支持 AMP、梯度累积、DDP reduce 与裁剪。
- 标签生成（必须）
  - 在 `_episodes_to_window_samples(...)` 中：除构建 `image/state/action/pad/prompt` 外，额外附加 `label_o`（1 或 None）。
  - 在 `_collate_samples(...)` 中：将 `label_o` 聚合到 batch（例如 `batch['is_positive'] = None/torch.ones(...)`）。
- 配置化（建议）
  - `cfgrl.dropout_prob`（默认 0.1）
  - `cfgrl.label_source ∈ {episode_rloo, window_rloo, window_return}`
  - `cfgrl.label_threshold`（默认 0）

4) 配置（`ript-vla_ori/config/algo/pi0_cfg_rl.yaml`）
- 新增：
  - `cfgrl:`
    - `dropout_prob: 0.1`
    - `label_source: episode_rloo`  # 可切 window_rloo/window_return
    - `label_threshold: 0.0`
    - `use_prefix_dropout: false`   # 可选
  - `algo.policy.condition_mode: token`（建议与 CFGRL 贴近；也可用 bias 做消融）
- 移除/忽略：
  - 训练端的 `alpha_uncond`、`use_binary_advantage` 等加权相关字段。
- 推理：
  - 保留 `rollout.inference.cfg_scale` 或使用 env 变量 `PI0_CFG_SCALE`；默认 1.0。

5) 运行期与兼容
- 推理 env 变量（保留）
  - `PI0_CFG_SCALE`：指导 w；
  - 不再使用 `PI0_IS_POSITIVE`（除非做单分支推理实验）。
- 训练日志（建议）
  - 记录 `o=1/∅` 比例、dropout 命中率；
  - 记录 FM 损失随 epoch 变化；
  - 评估阶段扫 `cfg_scale∈{1.0,1.5,2.0,3.0}` 的 SR 曲线。

实施步骤（按优先级）
1. 模型：实现 `o=∅` 的“真无条件”（suffix 无条件，必要）；推理使用 `v_∅/v_1` 组合。
2. Runner：记录 `episode['rewards']/['dones']`；保持现有窗口化结构。
3. Optimizer：改为“无加权 + 单分支 + 条件 dropout”的 CFGRL 训练；新增 `label_o` 流转。
4. 配置：新增 `cfgrl.*` 字段；移除/忽略加权相关字段；默认 `condition_mode=token`。
5. 验证：
   - 基线：`cfg_scale=1.0`；
   - 引导：扫 `cfg_scale` 曲线检查单调性与最优段；
   - 消融：`dropout_prob ∈ {0.0, 0.05, 0.1, 0.2}`、`label_source` 切换、`condition_mode` 切换。

伪代码（关键片段）

训练（窗口级 CFGRL）
```python
# 给每个窗口生成标签 o
A_w = estimate_window_advantage(ep)     # episode_rloo/window_return/window_rloo
o = 1 if A_w >= threshold else None     # None 表示 ∅（无条件）
if o == 1 and rand() < dropout_prob:
    o = None

# 构建 batch
batch = {
  'image': {...}, 'state': s0, 'action': a_seq, 'action_is_pad': mask,
  'prompt': [task_desc], 'is_positive': (None if o is None else torch.ones(B, dtype=long))
}

# 单分支前向（FM损失）
loss, loss_dict = model.model.forward(batch)  # 内部根据 is_positive(None/1) 走不同条件路径
```

推理（CF 引导）
```python
v_uncond = predict_velocity(..., is_positive=None)
v_cond   = predict_velocity(..., is_positive=1)
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

风险与注意事项
- 若仅实现“后缀无条件”，未做“前缀（语言）dropout”，CF 引导仍有效但可能弱于严格 CF；可在稳定后再加入 prefix dropout。
- 使用 episode 级 RLOO 作为窗口标签时，信号较粗；若 SR 曲线对 w 不敏感，可尝试 window 级度量或 per‑step 奖励。
- `token` 与 `bias` 条件模式可能带来轻微差异（token 会改变序列长度/注意力结构）。如未特化训练，`bias` 作为过渡也可用。

验收标准
- 训练端：在不加权的 CFGRL 训练下，loss 收敛稳定，o=∅ 样本占比接近设定的 dropout 概率。
- 推理端：cfg_scale 从 1.0 增加时，SR 曲线在合理区间内提升后饱和/回落（存在最优 w）。
- 代码端：不依赖优势加权/α 超参；`is_positive=None` 真正触发“无条件”路径。

文件改动清单（参考路径）
- `ript-vla_ori/pi0/modeling_pi0.py`：无条件/条件逻辑；推理组合。
- `ript-vla_ori/ript/env_runner/pi0_libero_runner.py`：记录 per‑step 奖励/终止标志。
- `ript-vla_ori/ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py`：改为 CFGRL（无加权+单分支+dropout+标签流转）。
- `ript-vla_ori/config/algo/pi0_cfg_rl.yaml`：新增 `cfgrl.*` 配置；整理旧字段。
- （可选）prefix dropout：`prepare_language`/prefix embed 增加 o=∅ 分支。

里程碑与时间评估（粗略）
- 第1天：模型与推理改造（真无条件/双分支融合）；Runner 补齐 per‑step 记录。
- 第2天：优化器改造（CFGRL 单分支 + 标签/Dropout）；配置联通；最小验证与日志。
- 第3天：CFG 扫描实验与消融；必要时加入 window 级优势估计或 prefix dropout。

