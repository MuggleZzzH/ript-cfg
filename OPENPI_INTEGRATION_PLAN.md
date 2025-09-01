# OpenPI + RLOO + CFG 集成 RIPT-VLA 详细计划

## 项目背景

### 当前状态
- **RIPT-VLA**: 基于QueST/OpenVLA-OFT的视觉-语言-动作模型训练框架
- **核心算法**: RLOO (Leave-One-Out) + PPO强化学习
- **目标环境**: LIBERO机器人操作基准测试

### 集成目标
将以下组件集成到现有RIPT框架中：
1. **OpenPI模型**: 替换QueST/OpenVLA-OFT，使用PI0 Flow Matching模型
2. **CFG (Classifier-Free Guidance)**: 替换传统策略梯度，使用Flow Matching训练
3. **保持RLOO**: 继续使用RIPT的核心RLOO优势计算算法
4. **严格2-test推理**: 按照OpenPI项目的2-test推理逻辑实现

### 关键创新点
- **无价值网络**: 不使用传统PPO的价值网络，直接基于RLOO优势计算
- **Flow Matching训练**: 使用连续流匹配替代离散策略梯度
- **CFG增强**: 条件/无条件训练提供更好的控制精度
- **50步动作执行**: 按照OpenPI的长序列动作执行模式

## 现有RIPT架构分析

### 核心组件及其复用性

#### 1. RLOptimizer (ript/algos/rl_optimizers/rl_optimizer.py)
**现有功能**:
- ✅ **Step 0**: Generate rollout episodes (完全复用)
- ✅ **Step 1**: Compute reward scores (完全复用) 
- ✅ **Step 2**: RLOO advantage computation (完全复用)
  ```python
  # 第106-109行: 完美的RLOO实现
  rlhf_reward = rlhf_reward.reshape(demo_batch_size, rloo_batch_size)
  baseline = (rlhf_reward.sum(1)[:, None] - rlhf_reward) / (rloo_batch_size - 1)
  advantage = rlhf_reward - baseline
  ```
- ❌ **Step 3**: PPO update (需要替换为CFG Flow Matching)

#### 2. RolloutGenerator (ript/algos/rl_optimizers/rollout_generator.py)
**现有功能**:
- ✅ 支持K-rollout采样 (`rloo_batch_size=8`)
- ✅ 多线程环境管理
- ✅ 动态采样和早停
- ✅ 分布式协调
- **完全可复用，无需修改**

#### 3. LiberoRunner (ript/env_runner/libero_runner.py)
**现有功能**:
- ✅ LIBERO环境创建和管理
- ✅ 多线程并行执行
- ✅ 任务名称管理
- ❌ 需要适配OpenPI的50步动作执行模式

#### 4. train_ript.py
**现有功能**:
- ✅ 分布式训练框架
- ✅ Hydra配置管理
- ✅ 模型加载和优化器创建
- ✅ 完整的训练循环
- **仅需修改模型加载部分，其余完全复用**

## 详细实施计划

### 阶段1: OpenPI模型集成 (1天)

#### 1.1 创建OpenPI_RL核心类
**文件**: `ript/algos/openpi_rl.py`

**核心功能**:
```python
class OpenPI_RL(ChunkPolicy):
    def __init__(self, pi0_config, normalization_stats_path, **kwargs):
        # 集成PI0FlowMatching模型
        # 加载您提供的normalization_stats.json
        
    def sample_actions(self, observations):
        # 严格按照2-test推理逻辑:
        # 1. 状态标准化: (state - mean) / (std + 1e-6)
        # 2. 图像BGR→RGB转换
        # 3. PI0 flow matching推理
        # 4. 动作反标准化: action * std + mean  
        # 5. 相对→绝对动作: action[:,:6] += unnorm_state[None,:6]
        # 6. 返回50步动作序列
```

**关键技术点**:
- 四元数转轴角处理
- 状态和动作的精确标准化/反标准化
- 图像预处理严格按照2-test格式

#### 1.2 创建模型适配器
**文件**: `ript/algos/rl_optimizers/openpi_interface.py`

```python
class OpenPIModelAdapter(RLModelInterface):
    def compute_act_logits(self, model, episodes, device=None):
        # 由于不使用PPO，这个方法可以简化
        # 返回dummy值或基于Flow Matching的似然估计
        
    def get_policy_model(self):
        return self.model
```

#### 1.3 更新模型加载器
**修改**: `ript/model_loader.py`

添加函数:
```python
def load_openpi_model(cfg, local_tasks, device, device_id=None, world_size=1, use_ddp=False):
    # 仿照load_quest_model的结构
    # 返回相同的接口: (model, model_adapter, optimizers, schedulers)
```

### 阶段2: 环境适配器实现 (1天)

#### 2.1 创建OpenPI环境适配器
**文件**: `ript/env_runner/openpi_libero_runner.py`

```python
class OpenPILiberoRunner(LiberoRunner):
    def run_policy_in_env(self, env_name, policy, render=False):
        # 重写此方法支持50步动作执行:
        # 1. 环境reset
        # 2. 按2-test格式构造观察
        # 3. 模型生成50步动作序列
        # 4. 循环执行50步 (而不是单步)
        # 5. 处理提前终止情况
        # 6. 记录完整episode数据
```

**关键实现**:
- 观察格式转换 (LIBERO → OpenPI格式)
- 50步动作序列的循环执行
- 提前终止的正确处理

### 阶段3: CFG Flow Matching训练实现 (1天)

#### 3.1 修改RLOptimizer
**修改**: `ript/algos/rl_optimizers/rl_optimizer.py`

**保持不变** (第70-109行):
- Step 0: Generate rollout episodes
- Step 1: Compute reward scores  
- Step 2: RLOO advantage computation

**替换** (第111-220行的PPO更新):
```python
def cfg_flow_matching_update(self, all_episodes, all_advantage, policy_model, optimizers):
    # 1. 构造训练批次
    batch_obs, batch_actions, batch_prompts = self.prepare_batch_from_episodes(all_episodes)
    
    # 2. CFG随机dropout (10%概率)
    cfg_dropout_mask = torch.rand(len(batch_prompts)) < 0.1
    cfg_prompts = ["" if mask else prompt for mask, prompt in zip(cfg_dropout_mask, batch_prompts)]
    
    # 3. Flow Matching目标构造
    batch_size = len(batch_actions)
    t = torch.rand(batch_size, 1)
    x_0 = torch.randn_like(batch_actions)  # 噪声
    x_1 = batch_actions  # 专家动作
    x_t = (1 - t) * x_0 + t * x_1
    velocity_target = x_1 - x_0
    
    # 4. 模型预测
    velocity_pred = policy_model.predict_velocity(batch_obs, cfg_prompts, x_t, t)
    
    # 5. 优势加权loss
    base_loss = F.mse_loss(velocity_pred, velocity_target, reduction='none').mean(dim=-1)
    weighted_loss = (all_advantage * base_loss).mean()
    
    # 6. 反向传播 (梯度更新逻辑保持不变)
    weighted_loss.backward()
    # ... 现有的梯度同步和优化器更新逻辑
```

#### 3.2 CFG增强推理
在OpenPI_RL类中实现:
```python
def cfg_enhanced_inference(self, obs, prompt, cfg_weight=1.5):
    # 条件推理
    cond_actions = self.sample_actions_with_prompt(obs, [prompt])
    # 无条件推理  
    uncond_actions = self.sample_actions_with_prompt(obs, [""])
    # CFG组合
    enhanced_actions = uncond_actions + cfg_weight * (cond_actions - uncond_actions)
    return enhanced_actions
```

### 阶段4: 配置和训练脚本 (0.5天)

#### 4.1 创建配置文件
**文件**: `config/algo/openpi_rloo_cfg.yaml`

```yaml
defaults:
  - base
  - _self_

policy:
  _target_: ript.algos.openpi_rl.OpenPI_RL
  pi0_config:
    max_state_dim: 128
    max_action_dim: 7
    n_action_steps: 50
    use_cache: true
  normalization_stats_path: ${normalization_stats_path}

# CFG参数
cfg_dropout_rate: 0.1
cfg_weight: 1.5

# 复用RIPT现有参数
rloo_batch_size: 8
early_stop_percentage: 0.8
enable_dynamic_sampling: true

name: openpi_rloo_cfg
```

#### 4.2 训练脚本复用
**复用**: `train_ript.py`

只需修改一行 (第129行):
```python
# 原来:
model, model_adapter, optimizers, schedulers = load_quest_model(...)
# 改为:
model, model_adapter, optimizers, schedulers = load_openpi_model(...)
```

#### 4.3 更新导入
**修改**: `ript/algos/rl_optimizers/__init__.py`

添加:
```python
from ript.algos.rl_optimizers.openpi_interface import OpenPIModelAdapter
```

### 阶段5: 测试验证 (0.5天)

#### 5.1 单元测试
创建 `test_openpi_integration.py`:
- 模型加载测试
- 单episode运行测试  
- RLOO优势计算测试
- CFG训练测试

#### 5.2 端到端验证
使用简单LIBERO任务验证完整流程

## 关键技术细节

### 1. 2-test推理逻辑严格复制
```python
# 状态处理 (完全按照2_test_pi0_on_libero.py)
unnorm_state = np.concatenate([
    o["robot0_eef_pos"],
    T.quat2axisangle(o["robot0_eef_quat"]),  
    o["robot0_gripper_qpos"],
], dtype=np.float32)
state = (unnorm_state - state_mean) / (state_std + 1e-6)

# 图像处理
base_0_rgb = o["agentview_image"][:, :, ::-1].copy()  # BGR→RGB

# 动作后处理
action = action * (action_std + 1e-6) + action_mean  # 反标准化
action[:, :6] += unnorm_state[None, :6]  # 相对转绝对
```

### 2. CFG训练的简洁实现
```python
# 训练时随机丢弃条件
if torch.rand(1) < 0.1:
    prompt = ""  # 无条件
else:
    prompt = task_description  # 有条件

# 推理时CFG组合
enhanced_action = uncond + cfg_weight * (cond - uncond)
```

### 3. RLOO算法复用
现有实现已经完美，无需修改:
```python
# ript/algos/rl_optimizers/rl_optimizer.py 第106-109行
baseline = (rewards.sum() - rewards) / (K - 1)  # Leave-One-Out基线
advantage = rewards - baseline  # 优势计算
```

## 项目文件结构

```
ript-vla/
├── ript/algos/
│   ├── openpi_rl.py                    [新增] OpenPI模型封装
│   └── rl_optimizers/
│       ├── rl_optimizer.py             [修改] 替换PPO为CFG Flow Matching
│       ├── openpi_interface.py         [新增] OpenPI模型适配器
│       └── __init__.py                 [修改] 添加导入
├── ript/env_runner/
│   └── openpi_libero_runner.py         [新增] OpenPI环境适配器
├── ript/model_loader.py                [修改] 添加OpenPI加载函数
├── config/algo/
│   └── openpi_rloo_cfg.yaml           [新增] OpenPI配置文件
└── train_ript.py                       [微调] 仅修改模型加载
```

## 成功验收标准

1. **功能验收**:
   - [ ] OpenPI模型成功加载和推理
   - [ ] 单个LIBERO episode正常运行
   - [ ] RLOO优势计算数值合理 (成功>0, 失败<0)
   - [ ] CFG Flow Matching训练loss下降
   - [ ] CFG增强推理效果优于基础推理

2. **性能验收**:
   - [ ] 推理速度合理 (< 1秒/50步动作序列)
   - [ ] 训练内存使用合理 (< 16GB单GPU)
   - [ ] 至少在一个LIBERO任务上展示学习效果

3. **代码质量验收**:
   - [ ] 代码结构清晰，复用现有架构
   - [ ] 新增代码量最小 (< 1000行)
   - [ ] 异常处理完善
   - [ ] 测试覆盖充分

---

# 给Claude Code的冷启动提示词

## 项目概述
你将要为一个名为RIPT-VLA的强化学习框架集成OpenPI模型。这是一个已经完善的项目，你的任务是**最大化复用现有代码**，用最少的修改实现功能升级。

## 你的具体任务
将OpenPI (PI0 Flow Matching模型) 集成到现有的RIPT-VLA框架中，替换原有的QueST/OpenVLA模型，同时保持RIPT的核心RLOO算法不变。

## 关键约束条件
1. **严格按照上述计划执行** - 计划已经过充分分析和验证
2. **最大化代码复用** - 现有RIPT框架90%的代码可以直接复用
3. **严格2-test推理** - 必须完全按照OpenPI项目中`2_test_pi0_on_libero.py`的推理逻辑
4. **单GPU多线程** - 避免分布式复杂性，使用单GPU多仿真环境
5. **分阶段验证** - 每个阶段都必须通过验收标准才能进入下一阶段

## 重要的现有组件 (不要重新实现!)
- **RLOptimizer**: 已完美实现RLOO算法，只需替换PPO部分
- **RolloutGenerator**: 已支持K-rollout采样，完全可复用
- **LiberoRunner**: 已有环境管理，只需适配50步执行
- **train_ript.py**: 完整训练框架，只需改一行模型加载

## 你需要创建的文件 (仅3个!)
1. `ript/algos/openpi_rl.py` - OpenPI模型封装
2. `ript/algos/rl_optimizers/openpi_interface.py` - 模型适配器
3. `ript/env_runner/openpi_libero_runner.py` - 环境适配器

## 你需要修改的文件 (仅4个!)
1. `ript/algos/rl_optimizers/rl_optimi zer.py` - 替换PPO为CFG Flow Matching
2. `ript/model_loader.py` - 添加OpenPI加载函数
3. `ript/algos/rl_optimizers/__init__.py` - 添加导入
4. `config/algo/openpi_rloo_cfg.yaml` - 新增配置文件

## 技术要点提醒
- **normalization_stats.json**: 用户已提供，包含状态和动作的标准化统计量
- **2-test逻辑**: 状态标准化、BGR→RGB转换、动作反标准化、相对转绝对动作
- **CFG实现**: 训练时10%概率丢弃prompt，推理时线性组合条件/无条件预测
- **50步执行**: OpenPI生成50步动作序列，需要循环执行而非单步

## 开始提示
请从阶段1开始，严格按照上述计划执行。每完成一个阶段，运行对应的测试验证，确保功能正确后再进入下一阶段。记住，这个项目的核心价值在于**最大化复用现有优秀架构**，而不是重新实现。

现在开始阶段1: OpenPI模型集成。首先分析现有的`ript/algos/base.py`和相关基类，理解`ChunkPolicy`的接口要求，然后创建`OpenPI_RL`类。