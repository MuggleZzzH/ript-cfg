## CFGRL（Classifier-Free Guidance RL）全面对齐改造方案（面向 ript-vla_ori）

本方案目标：在不重写现有 PI0 + Flow Matching（FM）框架的前提下，充分吸收 `cfgrl/ogbench/agents/cfgrl.py` 的训练与推理要点，实现“语义与行为等价”的对齐：
- 训练端：二分支（正向/无条件）FM 目标，支持小概率条件丢弃（classifier-free）与无条件权重 α；不在训练中使用 cfg 系数；支持优势门控（可二值/连续）。
- 推理端：严格使用双分支 CFG 合成公式 `v = v_uncond + cfg * (v_cond - v_uncond)`，并允许 test-time 调节 `cfg`；cfg 仅在推理生效。

### 一、参考实现小结（cfgrl.py）
- 行为流匹配（Behavioral Flow-Matching）损失：对 `x_t = (1−t)x_0 + t x_1` 预测速度 `vel = x_1 − x_0`，MSE。
- 无条件分支：训练时以概率 p（如 0.1）对条件做 dropout，用 `unc_embed` 替代 goals；实现“classifier-free”能力。
- 推理：循环 `flow_steps` 次，分别计算 `unc_vels` 与 `cond_vels`，按 `vels = unc + cfg*(cond−unc)` 合成，欧拉积分更新动作并裁剪到 [-1,1]。

对应到本仓库（PI0/FM，PyTorch+Transformer）应保留的等价性：
- 保持“正/无条件”两分支的并行/共享噪声与时间步采样；
- 训练不读取 cfg；
- 推理通过 cfg 混合两分支速度，支持 test-time 调节；
- 保证训练与推理“无条件”的语义一致（均为 is_positive=0 或使用 unc_embed 等价物）。

### 二、当前实现差异与改造要点
1) 条件注入位置与方式
- 现状：支持三种模式（bias/concat/token），默认 token；与 cfgrl.py 的 MLP 拼接不同，但语义可等价（尤其 concat 模式）。
- 要求：
  - 继续支持三模式，推荐评测默认 `token` 或 `concat`（更贴近原文“拼接”）。
  - 在 bias 模式下，推理时“无条件”必须显式 is_positive=0（而非 None），以免分布偏移。

2) 训练端“无条件”分支一致性
- 现状：优化器已分别前向 pos/uncond，并以 `alpha_uncond` 组合。
- 增强：加入小概率条件 dropout（如 p=0.1），即在训练批次中部分样本将 is_positive=0/或用占位 `unc_embed`；以增强无条件能力，靠近 cfgrl.py 的 `do_cfg` 逻辑。

3) 推理端 CFG 合成严格对齐
- 现状：token 模式已支持 `cfg_scale` 双分支；bias/concat 单分支默认可能走“无条件”。
- 要求：统一在 `sample_actions` 内实现两分支：
  - 计算 `v_uncond = f(obs, a, t, is_positive=0)` 与 `v_pos = f(obs, a, t, is_positive=1)`；
  - `v = v_uncond + cfg_scale * (v_pos - v_uncond)`；
  - 当 `cfg_scale==1.0` 时退化为 `v_pos`（保持与论文一致）。

4) 优势加权与损失权重
- 现状：支持二值优势（与 IQL Diffusion 一致）与 `alpha_uncond`；
- 建议：保持默认 `use_binary_advantage=true`、`alpha_uncond≈0.1`；如启用连续优势，需做归一化/温度缩放，避免权重失衡。

5) 时间步/噪声采样与欧拉积分方向
- 保持训练的连续 t 采样与推理欧拉积分方向/步长一致；不要引入与 FM 不兼容的随机扩散步长或温度缩放。

### 三、落地修改清单（最小必要改动）
1) `ript-vla_ori/pi0/modeling_pi0.py`
- `sample_actions`：无论 `condition_mode`，统一实现双分支 CFG 合成；当 `cfg_scale==1.0`，直接使用正向分支；当 `cfg_scale!=1.0`，计算 `v_uncond + cfg*(v_pos - v_uncond)`；确保 `is_positive` 参数可显式传入 0/1。
- `predict_velocity/forward`：推理时支持 `is_positive` 明确取值，bias 模式也使用嵌入向量（非 None）。
- 保持三模式（bias/concat/token）实现不变，仅修正 token 前缀掩码（`att_masks[:, :2] = True`）。

2) `ript-vla_ori/ript/algos/rl_optimizers/cfg_flow_optimizer_pi0.py`
- 在构造训练批时，以小概率执行条件 dropout：对部分样本将 `is_positive` 置 0 或替换为无条件嵌入（如实现），与 cfgrl.py 的 `do_cfg` 一致；
- 继续保持正/无条件分支共享 `noise/time` 的 FM 损失计算，并按 `L = w_pos * L_pos + α * L_uncond` 合成。

3) `ript-vla_ori/ript/algos/policies/pi0_oft_policy.py`
- `select_action` 透传 `cfg_scale` 与 `is_positive_infer` 到模型 `sample_actions`；当 `cfg_scale==1.0` 时，明确走正向分支。

4) `ript-vla_ori/config/algo/pi0_cfg_rl.yaml`
- 保持：
  - `policy.condition_mode: token`（或 `concat` 以更贴近论文拼接语义）。
  - `inference.cfg_scale` 和 `inference.is_positive_infer`；默认 `cfg_scale: 1.0`，可由环境变量 `PI0_CFG_SCALE` 覆盖。

### 四、接口与使用说明
- 训练：不使用 cfg；可选择是否启用条件 dropout（默认 p=0.1）。
- 推理：
  - 传入 `cfg_scale`（>1 可增强策略倾向），与 `is_positive_infer`（若需强制单分支）。
  - 推荐：起步 `cfg_scale ∈ [1.5, 3.0]`，逐步微调。

### 五、风险与验证
- 若模型未经过“无条件分支”充分训练，直接启用双分支 CFG 可能造成动作抖动或性能下降；需先稳定单分支（cfg=1.0，走正向分支），再逐步增大 cfg。
- 确保训练/推理对“无条件”的定义一致（is_positive=0），避免 bias 模式下的 None 语义偏移。

### 六、对齐结论
- 本方案在不重写为 JAX/Flax 的前提下，严格对齐了 cfgrl.py 的“训练不读 cfg、推理双分支混合、条件 dropout”的核心思想；
- 通过在 PI0/FM 框架中统一双分支合成、显式无条件语义、可选条件 dropout，实现训练与推理路径的功能等价。

### 优势的作用域与使用（chunk/episode 级）
- 现状约束：我们一次生成的是整段 action chunk（序列），同一条轨迹/其派生的所有窗口 chunk 共享同一个优势标量（episode-level）。
- 与 CFGRL 的关系：
  - 训练端（对齐 cfgrl.py）：优势不进入损失加权；仅做纯 MSE（FM 目标）+ 条件 dropout（p_dropout_cfg）。
  - 推理端：优势不参与 CFG；仅通过 `cfg` 在无条件/有条件速度场间插值。
- 合理用法（在不打破 CFGRL 设定的前提下）：
  - 过滤/采样：可选地仅保留优势≥阈值的 chunk 进入训练（positive-only 训练集），或按优势做数据重采样；默认关闭。
  - 日志/评估：在窗口级训练日志中携带 `chunk_advantage` 作为元数据（用于对照学习曲线与成功率），不参与反向传播。
- 配置建议：
  - `algo.cfgrl.use_advantage_in_training: false`（默认，严格对齐 cfgrl.py）。
  - `algo.cfgrl.filter_positive_chunks: false`、`algo.cfgrl.adv_threshold: 0.0`（可选过滤）。
  - `algo.cfgrl.p_dropout_cfg: 0.1`（训练期条件 dropout 概率）。

### 前三项详细改造建议（代码级）

以下建议直接面向当前代码结构，若与现有实现冲突，一律以本对齐方案为准进行替换。

1) 新增 Torch 版 CFGRL Agent（单网 + 条件 dropout）
- 新文件：`ript-vla_ori/ript/algos/rl_optimizers/cfgrl_agent_torch.py`
```python
import torch
import torch.nn as nn
from typing import Dict, Any

class CFGRLAgentTorch(nn.Module):
    def __init__(self, model, flow_steps: int = 16, p_dropout_cfg: float = 0.1):
        super().__init__()
        self.model = model  # 需支持 predict_velocity_gc(...) 与 get_unc_goals(...)
        self.flow_steps = flow_steps
        self.p_dropout_cfg = p_dropout_cfg

    @torch.no_grad()
    def sample_actions_cfgrl(self, observations: Dict[str, Any], cfg: float = 3.0) -> torch.Tensor:
        """双分支 CFG 采样：v = v_uncond + cfg*(v_cond - v_uncond)"""
        device = next(self.model.parameters()).device
        # 1) 准备 obs 与初始动作噪声
        images = observations["images"]  # (B, ...)
        img_masks = observations.get("img_masks", None)
        lang_tokens = observations["lang_tokens"]
        lang_masks = observations["lang_masks"]
        state = observations["state"]  # (B, S)

        bsize = state.shape[0]
        actions = torch.randn(
            (bsize, self.model.config.n_action_steps, self.model.config.max_action_dim),
            device=device, dtype=state.dtype
        )
        # 2) 编码 goals
        cond_goals = self.model.encode_goals(lang_tokens, lang_masks)         # (B, D)
        uncond_goals = self.model.get_unc_goals(bsize)                        # (B, D)

        # 3) 前缀缓存（视觉+语言），减少重复计算
        prefix = self.model.build_prefix_cache(images, img_masks, lang_tokens, lang_masks)

        # 4) 欧拉步进
        for i in range(self.flow_steps):
            t = torch.full((bsize,), float(i) / self.flow_steps, device=device, dtype=state.dtype)
            v_uncond = self.model.predict_velocity_gc(prefix, state, actions, t, goals=uncond_goals)
            v_cond   = self.model.predict_velocity_gc(prefix, state, actions, t, goals=cond_goals)
            v        = v_uncond + cfg * (v_cond - v_uncond)
            actions  = actions + v / self.flow_steps

        return actions.clamp_(-1.0, 1.0)

    def actor_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """FM MSE + 条件 dropout（训练不读取 cfg）"""
        device = next(self.model.parameters()).device
        actions = batch["actions"]                     # x_1 (B, T, A)
        bsize   = actions.shape[0]
        noise   = torch.randn_like(actions)             # x_0 ~ N(0,I)
        # 采样 t ∈ [0,1]
        t = torch.rand((bsize,), device=device, dtype=actions.dtype)
        x_t = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * actions
        vel = actions - noise

        # 条件构造：p_dropout_cfg 概率替换为无条件 goals
        cond_goals = self.model.encode_goals(batch["lang_tokens"], batch["lang_masks"])  # (B,D)
        uncond_goals = self.model.get_unc_goals(bsize)                                     # (B,D)
        do_uncond = (torch.rand((bsize,), device=device) < self.p_dropout_cfg)
        goals = torch.where(do_uncond.view(-1, 1), uncond_goals, cond_goals)

        # 统一的前缀缓存
        prefix = self.model.build_prefix_cache(batch["images"], batch.get("img_masks", None),
                                               batch["lang_tokens"], batch["lang_masks"]) 
        pred = self.model.predict_velocity_gc(prefix, batch["state"], x_t, t, goals=goals)
        loss = torch.mean((pred - vel) ** 2)
        return loss
```
- 说明：
  - 该 Agent 仅依赖模型提供的 3 个新 API：`encode_goals(...)`、`get_unc_goals(B)`、`predict_velocity_gc(prefix, state, x_t, t, goals)`；以及 `build_prefix_cache(...)` 用于一次前缀编码复用多步。
  - 训练仅做 MSE + 条件 dropout，不读取优势与 cfg；推理始终双分支 CFG 混合。

2) 在 PI0 模型增加 unc_embed（可学习无条件嵌入）并替换 is_positive → goals
- 修改：`ript-vla_ori/pi0/modeling_pi0.py`
  - 在 `__init__`：新增无条件嵌入参数与语言→goal 投影层。
```python
# 在 __init__ 内新增
self.unc_embed = nn.Parameter(torch.zeros(self.config.proj_width))  # (D)
# 使用文本 token 嵌入维度自动适配
text_emb_dim = self.paligemma_with_expert.get_input_embeddings().embedding_dim
self.goal_from_lang = nn.Sequential(
    nn.Linear(text_emb_dim, self.config.proj_width),
    nn.GELU(),
    nn.Linear(self.config.proj_width, self.config.proj_width),
)
```
  - 新增无条件目标构造：
```python
def get_unc_goals(self, bsize: int) -> torch.Tensor:
    return self.unc_embed.unsqueeze(0).expand(bsize, -1)
```
  - 新增语言 pooling → goals：
```python
@torch.no_grad()
def encode_goals(self, lang_tokens: torch.Tensor, lang_masks: torch.Tensor) -> torch.Tensor:
    # lang_tokens: (B, L), lang_masks: (B, L) ，True=valid
    tok_embed = self.paligemma_with_expert.get_input_embeddings()(lang_tokens)  # (B,L,E)
    mask = lang_masks.to(tok_embed.dtype)
    pooled = (tok_embed * mask.unsqueeze(-1)).sum(dim=1) / mask.clamp_min(1e-6).sum(dim=1, keepdim=True)
    return self.goal_from_lang(pooled)  # (B, D=proj_width)
```
  - 统一前缀缓存（减少多步重复编码）：
```python
@torch.no_grad()
def build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    # 计算 KV cache 供后续多步复用
    _, past_kv = self.paligemma_with_expert.forward(
        attention_mask=att_2d, position_ids=pos_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=self.config.use_cache, fill_kv_cache=True,
    )
    return {"prefix_pad_masks": prefix_pad_masks, "att_2d": att_2d, "pos_ids": pos_ids, "past_kv": past_kv}
```
  - 用 goals 替换 is_positive：
    - 修改 `embed_suffix(..., is_positive=None)` → `embed_suffix(..., goals: torch.Tensor | None)`；当 `goals is None` 使用 `get_unc_goals(B)`。
    - 注入规则：
      - bias：`action_time_emb += goals.unsqueeze(1)`
      - concat：`cat([action_time_emb, goals.unsqueeze(1).expand(-1,T,-1)], dim=-1) → Linear(..., proj_width)`
      - token：`suffix = torch.cat([goals.unsqueeze(1), state_token, action_time_tokens], dim=1)` 并修正 `att_masks[:, :2] = True`

3) 实现观测与目标编码：language pooling 作为 goals（训练/推理统一）
- 训练侧：`CFGRLAgentTorch.actor_loss` 中调用 `model.encode_goals(lang_tokens, lang_masks)` 得到 `cond_goals`；按 p_dropout 替换为 `get_unc_goals(B)`；前向走 `predict_velocity_gc(...)`。
- 推理侧：`CFGRLAgentTorch.sample_actions_cfgrl` 中同时构造 `cond_goals` 与 `uncond_goals`，按 CFG 合成速度场。
- 模型侧 `predict_velocity_gc` 实现建议（在 `modeling_pi0.py` 内新增）：
```python
@torch.no_grad()
def predict_velocity_gc(self, prefix, state, x_t, t, goals: torch.Tensor) -> torch.Tensor:
    # prefix: 来自 build_prefix_cache 的缓存（含 past_kv/att_2d/pos_ids）
    # 1) 构造后缀（以 goals 注入代替 is_positive）
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
        state=state, noisy_actions=x_t, timestep=t, goals=goals
    )
    # 2) 组合 2D 注意力与位置 id
    pad_masks = torch.cat([prefix["prefix_pad_masks"], suffix_pad_masks], dim=1)
    att_2d = make_att_2d_masks(pad_masks, torch.cat([torch.ones_like(prefix["prefix_pad_masks"]), suffix_att_masks], dim=1))
    pos_ids = torch.cumsum(pad_masks, dim=1) - 1
    # 3) 前向得到最后 n_action_steps 的速度场预测
    outs, _ = self.paligemma_with_expert.forward(
        attention_mask=att_2d, position_ids=pos_ids, past_key_values=prefix["past_kv"],
        inputs_embeds=[None, suffix_embs], use_cache=False, fill_kv_cache=False,
    )
    return self.action_out_proj(outs[:, -self.config.n_action_steps:])  # (B, T, A)
```

> 备注：以上 3 点完成后，即构成了“单网 + 条件 dropout”的训练与“双分支 CFG”的推理闭环；不再使用 `is_positive`，优势仅做数据筛选/日志（不入损失），`cfg` 仅推理生效。


