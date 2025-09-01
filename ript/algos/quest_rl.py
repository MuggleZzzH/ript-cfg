import os
import fcntl
import uuid
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import itertools
import copy
import time
from collections import deque
from tqdm import tqdm

import ript.utils.tensor_utils as TensorUtils
import ript.utils.obs_utils as ObsUtils
from ript.utils.utils import map_tensor_to_device
from ript.algos.base import ChunkPolicy
from ript.algos.quest import QueST

###############################################################################
# Global counter helper used to synchronize rollout generation across processes.
###############################################################################
class FileGlobalCounter:
    def __init__(self, filename):
        self.filename = filename
        # Create the file if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write("0")
    
    def reset(self, value=0):
        with open(self.filename, "w") as f:
            f.write(str(value))
    
    def update(self, increment=1):
        with open(self.filename, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
                current += increment
                f.seek(0)
                f.write(str(current))
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current
    
    def get(self):
        with open(self.filename, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current

###############################################################################
# Merged RL class that combines RL optimization with action sampling.
###############################################################################
class QueST_rl(QueST):
    """
    This class uses RL to finetune the single-step Quest policy.
    It combines rollout generation and PPO-based optimization routines with RL-specific
    action sampling. Distributed operations (DDP) are safely guarded so that the code
    can also run without DDP.
    """
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 stage,
                 loss_fn,
                 l1_loss_scale,
                 **kwargs):
        # 设置无效的log概率的默认值
        self.INVALID_LOGPROB = 1.0
        # 调用父类的初始化方法，传递所有参数
        super().__init__(autoencoder, policy_prior, stage, loss_fn, l1_loss_scale, **kwargs)

    ##############################################################
    # 动作采样方法（来自原始的QueST_rl代码）。
    ##############################################################
    def sample_actions(self, data):
        # 对输入数据进行预处理，train_mode=False表示推理模式
        data = self.preprocess_input(data, train_mode=False)
        # 获取上下文信息
        context = self.get_context(data)
        # 使用policy_prior根据上下文获取top-k的动作索引
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        # 通过autoencoder将动作索引解码为实际动作
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        # 调整动作张量的维度顺序
        pred_actions = pred_actions.permute(1, 0, 2)

        # 将预测的动作从tensor转为numpy数组，并移到CPU
        pred_actions = pred_actions.detach().cpu().numpy()
        # 将上下文token转为numpy数组
        context_tokens = context.detach().cpu().numpy()  # (B, L_obs, D)
        # 将动作索引转为numpy数组
        action_indices = sampled_indices.detach().cpu().numpy()  # (B, L_act)
        
        # 返回预测动作、上下文token和动作索引
        return pred_actions, context_tokens, action_indices

    def get_action(self, obs, task_id, task_emb=None):
        """
        从动作队列中获取一个动作。如果队列为空，则处理观测、采样动作并补充队列。
        """
        # 确保action_queue已初始化，否则抛出异常
        assert hasattr(self, "action_queue") and self.action_queue is not None, "you need to call policy.reset() before getting actions"

        # 设置模型为评估模式
        self.eval()
        # 如果动作队列为空，需要重新采样动作
        if len(self.action_queue) == 0:
            # 遍历观测obs中的每个键值对
            for key, value in obs.items():
                # 如果是图像类型的观测，进行图像预处理
                if key in self.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                # 如果是低维观测，转换为float类型
                elif key in self.lowdim_encoders:
                    value = TensorUtils.to_float(value)
                # 将处理后的观测转为tensor
                obs[key] = torch.tensor(value)
            # 构造batch字典
            batch = {}
            batch["obs"] = obs
            # 如果提供了任务嵌入，则加入batch
            if task_emb is not None:
                batch["task_emb"] = task_emb
            # 否则使用任务id
            else:
                batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
            # 将batch中的tensor移动到指定设备
            batch = map_tensor_to_device(batch, self.device)
            # 在不计算梯度的上下文中采样动作
            with torch.no_grad():
                pred_actions, context_tokens, action_indices = self.sample_actions(batch)
                # 将采样到的动作填充到动作队列中，只取action_horizon步
                self.action_queue.extend(pred_actions[:self.action_horizon])
        else:
            # 如果队列不为空，则上下文和动作索引设为None
            context_tokens = None
            action_indices = None
        
        # 从动作队列中弹出一个动作
        action = self.action_queue.popleft()
        # 返回动作、上下文token和动作索引
        return action, context_tokens, action_indices
