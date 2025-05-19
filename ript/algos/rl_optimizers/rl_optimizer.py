import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm


class RLOptimizer:
    """
    RL optimizer that implements PPO algorithm for policy optimization.
    Simplified version based on Quest_rl implementation.
    """
    
    def __init__(
        self,
        rollout_generator,
        reward_function,
        ppo_clip_range=0.2,
        num_ppo_epochs=5,
        ppo_batch_size=64,
        gradient_accumulation_steps=1,
        grad_norm_clip=None,
        use_minibatch_permutation=True,
        ppo_clip_high=0.2,
        use_token_level_loss_avg=True
    ):
        """
        Initialize the RL optimizer.
        
        Args:
            rollout_generator: RolloutGenerator instance for generating rollouts
            reward_function: Reward function to compute rewards
            ppo_clip_range: PPO clipping range parameter
            num_ppo_epochs: Number of PPO epochs per optimization
            ppo_batch_size: PPO batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            grad_norm_clip: Gradient norm clipping value
            use_minibatch_permutation: Whether to use minibatch permutation
            ppo_clip_high: Upper limit for PPO clipping
            use_token_level_loss_avg: Whether to use token-level loss averaging
        """
        self.rollout_generator = rollout_generator
        self.reward_function = reward_function
        self.ppo_clip_range = ppo_clip_range
        self.num_ppo_epochs = num_ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_norm_clip = grad_norm_clip
        self.use_minibatch_permutation = use_minibatch_permutation
        self.ppo_clip_high = ppo_clip_high
        self.use_token_level_loss_avg = use_token_level_loss_avg
        
        # For invalid log probabilities
        self.INVALID_LOGPROB = 1.0
    
    def optimize(self, model_interface, batch, optimizers, data_iterator=None, dataloader=None):
        """
        Optimize the policy using PPO with early stopping rollout generation.
        
        Args:
            model_interface: Model interface for policy model
            batch: Batch of data for optimization
            optimizers: List of optimizers to use
            data_iterator: Iterator for data batches (optional)
            dataloader: Dataloader for reinitializing data iterator (optional)
            
        Returns:
            dict: Optimization metrics
        """
        with torch.no_grad():
            # Step 0: Generate rollout episodes
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f'rank {rank} start generate rollout episodes...')
            import time
            start_time = time.time()
            
            # Generate rollouts - now returns episodes, task_ids, valid_mask, samples_checked
            all_episodes, all_task_ids, valid_mask, samples_checked = self.rollout_generator.generate_rollouts(
                model_interface.model, batch, data_iterator, dataloader
            )
            
            print(f'rank {rank} finished generate rollout episodes in {time.time() - start_time:.2f} seconds')
            
            # Step 1: Compute reward scores
            all_scores = []
            for i in range(len(all_episodes)):
                reward = self.reward_function.compute_reward(i, all_episodes[i], batch)
                all_scores.append(reward)
            all_scores = torch.tensor(all_scores).to(model_interface.device)
            
            # Step 2: Compute advantage
            policy_model = model_interface.get_policy_model()

            # Compute log probs
            model_logprob, max_seq_len = model_interface.compute_act_logits(
                policy_model, all_episodes, device=model_interface.device
            )
            
            rlhf_reward = all_scores
            
            # Reshape rewards for advantage computation
            # Get batch sizes from the rollout generator
            demo_batch_size = len(all_episodes) // self.rollout_generator.rloo_batch_size
            rloo_batch_size = self.rollout_generator.rloo_batch_size
            
            rlhf_reward = rlhf_reward.reshape(demo_batch_size, rloo_batch_size)
            baseline = (rlhf_reward.sum(1)[:, None] - rlhf_reward) / (rloo_batch_size - 1)
            advantage = rlhf_reward - baseline
            all_advantage = advantage.flatten()
        
        # Step 3: Optimize with PPO
        ppo_dataset_size = len(all_episodes)
        gradient_step = 0
        
        pg_stats = []
        pg_ratio_stats = []
        pg_clipfrac_stats = []
        
        for epoch in tqdm(range(self.num_ppo_epochs), desc='PPO Epochs'):
            if self.use_minibatch_permutation:
                b_inds = np.random.permutation(ppo_dataset_size)
            else:
                b_inds = np.arange(ppo_dataset_size)
            
            for mini_batch_start in range(0, ppo_dataset_size, self.ppo_batch_size):
                mini_batch_end = min(mini_batch_start + self.ppo_batch_size, ppo_dataset_size)
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                
                mb_advantage = all_advantage[mini_batch_inds]
                mb_episodes = [all_episodes[i] for i in mini_batch_inds]
                mb_model_logprob = model_logprob[mini_batch_inds]
                mb_valid_mask = valid_mask[mini_batch_inds]
                
                # Compute new logprobs (requires gradient)
                new_logprob, _ = model_interface.compute_act_logits(
                    policy_model, mb_episodes, device=model_interface.device, max_seq_len=max_seq_len
                )
                
                # Process episodes to get action token mask for loss computation
                _, _, mb_action_token_mask, _ = model_interface.process_episodes(
                    mb_episodes, device=model_interface.device, max_seq_len=max_seq_len
                )
                
                # Compute probability ratios
                new_ratio = (new_logprob - mb_model_logprob).exp()
                
                # Compute losses
                pg_losses = -mb_advantage[:, None] * new_ratio
                pg_losses2 = -mb_advantage[:, None] * torch.clamp(
                    new_ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_high
                )
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                
                # Get token step mask and ratio
                B, T, A = mb_action_token_mask.shape
                token_step_mask = mb_action_token_mask.view(B, T * A)
                ratio = new_ratio.mean(-1)
                
                # Compute loss based on token level averaging
                if self.use_token_level_loss_avg:
                    mb_token_step_valid_mask = mb_valid_mask.unsqueeze(1).repeat(1, token_step_mask.shape[1])
                    mb_token_step_valid_mask = mb_token_step_valid_mask & token_step_mask
                    pg_loss_max = pg_loss_max.masked_fill(~mb_token_step_valid_mask, 0.0)
                    pg_loss = pg_loss_max.sum() / (mb_token_step_valid_mask.sum() + 1e-6)
                else:
                    pg_loss_max = pg_loss_max.masked_fill(~token_step_mask, 0.0)
                    pg_loss_max = pg_loss_max.sum(1) / token_step_mask.sum(1)
                    pg_loss_max = pg_loss_max.masked_fill(~mb_valid_mask, 0.0)
                    pg_loss = pg_loss_max.mean()
                
                # Backpropagate
                pg_loss.backward()
                
                # Synchronize gradients in distributed setting
                if dist.is_initialized():
                    for param in policy_model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                
                gradient_step += 1
                
                # Update parameters if gradient accumulation steps reached
                if gradient_step == self.gradient_accumulation_steps:
                    if self.grad_norm_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            policy_model.parameters(), self.grad_norm_clip
                        )
                    
                    for optimizer in optimizers:
                        optimizer.step()
                    
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    
                    gradient_step = 0
                
                # Collect statistics
                with torch.no_grad():
                    pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                    pg_clipfrac_stats.append(pg_clipfrac.detach().item())
                    pg_stats.append(pg_loss.detach().item())
                    pg_ratio_stats.append(ratio.mean().detach().item())
        
        # Step 4: Collect and return metrics
        metrics = {
            'pg_clipfrac_stats': torch.tensor(pg_clipfrac_stats).mean(),
            'pg_loss_stats': torch.tensor(pg_stats).mean(),
            'pg_ratio_stats': torch.tensor(pg_ratio_stats).mean(),
            'mean_scores': torch.tensor(all_scores[valid_mask]).mean(),
            'mean_advantage': torch.tensor(all_advantage[valid_mask]).mean(),
            'mean_rlhf_reward': torch.tensor(rlhf_reward).mean(),
            'rollout_checked': torch.tensor(samples_checked)
        }
        
        if self.grad_norm_clip is not None:
            metrics.update({'grad_norm': grad_norm})
        
        # Reduce metrics across devices in distributed setting
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                if dist.is_initialized():
                    metrics[key] = metrics[key].to(model_interface.device)
                    dist.all_reduce(metrics[key], op=dist.ReduceOp.AVG)
                metrics[key] = metrics[key].cpu().item()
        
        # Collect task-specific success rates
        task_srs = {}
        task_ids_in_batch = torch.unique(all_task_ids).tolist()
        for task_id in task_ids_in_batch:
            task_name = self.rollout_generator.task_names_to_use[task_id]
            task_mask = all_task_ids == task_id
            if task_mask.any():
                valid_task_mask = valid_mask & task_mask
                task_srs[f'rl_train_succeess_rate/{task_name}'] = torch.tensor(all_scores)[valid_task_mask].mean().item()
        
        # Gather task success rates in distributed setting
        if dist.is_initialized():
            all_task_srs = [None] * dist.get_world_size()
            dist.all_gather_object(all_task_srs, task_srs)
            
            if dist.get_rank() == 0:
                for task_sr in all_task_srs:
                    for key, value in task_sr.items():
                        if key not in metrics:
                            metrics[key] = value
                        else:
                            metrics[key] = (metrics[key] + value) / 2.0
        else:
            metrics.update(task_srs)
        
        return metrics 