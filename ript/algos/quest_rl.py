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
        self.INVALID_LOGPROB = 1.0
        super().__init__(autoencoder, policy_prior, stage, loss_fn, l1_loss_scale, **kwargs)

    ##############################################################
    # Action sampling methods (from the original QueST_rl code).
    ##############################################################
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        pred_actions = pred_actions.permute(1, 0, 2)

        pred_actions = pred_actions.detach().cpu().numpy()
        context_tokens = context.detach().cpu().numpy()  # (B, L_obs, D)
        action_indices = sampled_indices.detach().cpu().numpy()  # (B, L_act)
        
        return pred_actions, context_tokens, action_indices

    def get_action(self, obs, task_id, task_emb=None):
        """
        Retrieves an action from the action queue. If the queue is empty, processes the observation,
        samples actions, and refills the queue.
        """
        assert hasattr(self, "action_queue") and self.action_queue is not None, "you need to call policy.reset() before getting actions"

        self.eval()
        if len(self.action_queue) == 0:
            for key, value in obs.items():
                if key in self.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                elif key in self.lowdim_encoders:
                    value = TensorUtils.to_float(value)
                obs[key] = torch.tensor(value)
            batch = {}
            batch["obs"] = obs
            if task_emb is not None:
                batch["task_emb"] = task_emb
            else:
                batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
            batch = map_tensor_to_device(batch, self.device)
            with torch.no_grad():
                pred_actions, context_tokens, action_indices = self.sample_actions(batch)
                self.action_queue.extend(pred_actions[:self.action_horizon])
        else:
            context_tokens = None
            action_indices = None
        
        action = self.action_queue.popleft()
        return action, context_tokens, action_indices
