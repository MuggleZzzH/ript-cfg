import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch, laplace_log_prob
import gc
def transpose_kv_chunk(prompt_key_values_chunk):
    num_layers = len(prompt_key_values_chunk[0])
    layerwise = [[] for _ in range(num_layers)]
    for t_local in range(len(prompt_key_values_chunk)):
        for layer in range(num_layers):
            layerwise[layer].append(prompt_key_values_chunk[t_local][layer])
    return layerwise

ACTION_DIM = 7

class RLOptimizerOpenVLAOFT:
    """
    RL optimizer that implements PPO algorithm for policy optimization.
    Simplified version based on Quest_rl implementation.
    """
    
    def __init__(
        self,
        rollout_generator,
        reward_function,
        ppo_clip_range=0.2,
        ppo_clip_high=0.2,
        num_ppo_epochs=1,
        gradient_accumulation_steps=1,
        grad_norm_clip_model=None,
        grad_norm_clip_header=None,
        tokens_per_step=8,
        max_step_batch_size=4,
        rloo_over_all_rollouts=False,
        log_prob_mode='sum_on_action_dim',
    ):
        """
        Initialize the RL optimizer.
        
        Args:
            rollout_generator: RolloutGenerator instance for generating rollouts
            reward_function: Reward function to compute rewards
            ppo_clip_range: PPO clipping range parameter
            kl_coef: KL divergence coefficient
            num_ppo_epochs: Number of PPO epochs per optimization
            ppo_batch_size: PPO batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            per_token_kl: Whether to use per-token KL divergence
            grad_norm_clip: Gradient norm clipping value
            use_minibatch_permutation: Whether to use minibatch permutation
            ppo_clip_high: Upper limit for PPO clipping
            use_token_level_loss_avg: Whether to use token-level loss averaging
        """
        self.rollout_generator = rollout_generator
        self.reward_function = reward_function
        self.ppo_clip_range = ppo_clip_range
        self.num_ppo_epochs = num_ppo_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_norm_clip_model = grad_norm_clip_model
        self.grad_norm_clip_header = grad_norm_clip_header
        self.ppo_clip_high = ppo_clip_high
        self.tokens_per_step = tokens_per_step
        self.max_step_batch_size = max_step_batch_size
        self.rloo_over_all_rollouts = rloo_over_all_rollouts
        self.log_prob_mode = log_prob_mode # 'sum_on_action_dim' or 'avg_on_action_dim' or 'ratio_on_action_dim'
    def optimize(self, model, batch, optimizers, data_iterator=None, dataloader=None):
        """
        Optimize the policy using PPO with early stopping rollout generation.
        
        Args:
            model: Model interface for policy and reference models
            batch: Batch of data for optimization
            data_iterator: Iterator for data batches (optional)
            dataloader: Dataloader for reinitializing data iterator (optional)
            
        Returns:
            dict: Optimization metrics
        """
        with torch.no_grad():
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f'rank {rank} start generate rollout episodes...')
            import time
            start_time = time.time()

            # Generate rollouts - now returns episodes, task_ids, valid_mask, samples_checked
            all_episodes, all_task_ids, valid_mask, samples_checked = self.rollout_generator.generate_rollouts(
                model, batch, data_iterator, dataloader
            )
            
            print(f'rank {rank} finished generate rollout episodes in {time.time() - start_time:.2f} seconds')
            
            # Step 1: Compute reward scores
            all_scores = []
            for i in range(len(all_episodes)):
                reward = self.reward_function.compute_reward(i, all_episodes[i], batch)
                all_scores.append(reward)
            rlhf_reward = torch.tensor(all_scores).to(model.device)
            
            demo_batch_size = len(all_episodes) // self.rollout_generator.rloo_batch_size
            rloo_batch_size = self.rollout_generator.rloo_batch_size
            
            if self.rloo_over_all_rollouts:
                rlhf_reward = rlhf_reward.reshape(1, demo_batch_size*rloo_batch_size)
                devider = demo_batch_size*rloo_batch_size - 1
            else:
                rlhf_reward = rlhf_reward.reshape(demo_batch_size, rloo_batch_size)
                devider = rloo_batch_size - 1
            baseline = (rlhf_reward.sum(1)[:, None] - rlhf_reward) / devider
            advantage = rlhf_reward - baseline
            all_advantage = advantage.flatten()
        
        # Step 3: Optimize with PPO
        pg_clipfrac_stats = []
        pg_loss_stats = []
        pg_ratio_stats = []
        gradient_norm_model_stats = []
        gradient_norm_header_stats = []

        cfg = model.cfg
        processor = model.processor
        vla = model.model.module if hasattr(model.model, 'module') else model.model
        action_head = model.action_head.module if hasattr(model.action_head, 'module') else model.action_head
        proprio_projector = model.proprio_projector.module if hasattr(model.proprio_projector, 'module') else model.proprio_projector
        noisy_action_projector = None
        scale_head = model.scale_head.module if hasattr(model.scale_head, 'module') else model.scale_head
        
        for epoch in tqdm(range(self.num_ppo_epochs), desc='PPO Epochs'):
            gradient_step = 0
            episide_random_idx = torch.randperm(len(all_episodes))
            print('episode_random_idx', episide_random_idx)
            for bidx in tqdm(episide_random_idx, desc='PPO Batch'):
                num_valid_step = sum(all_episodes[bidx]['valid'][i] for i in range(len(all_episodes[bidx]['valid'])))
                
                num_total_tokens = num_valid_step * self.tokens_per_step
                if self.log_prob_mode == 'ratio_on_action_dim':
                    num_total_tokens *= ACTION_DIM
                
                valid_value_count = 0

                task_description = all_episodes[bidx]['task_description'][0]

                for start_tidx in range(0, num_valid_step, self.max_step_batch_size):
                    end_tidx = min(start_tidx + self.max_step_batch_size, num_valid_step-1)

                    if end_tidx <= start_tidx:
                        end_tidx = start_tidx + 1

                    batch_ref_logprob = torch.stack([torch.tensor(all_episodes[bidx]['log_prob'][tidx]).to(model.device) for tidx in range(start_tidx, end_tidx)])
                    batch_ref_action_normalized = torch.stack([torch.tensor(all_episodes[bidx]['actions_normalized'][tidx]).to(model.device) for tidx in range(start_tidx, end_tidx)])

                    batch_obs_input = all_episodes[bidx]['observations'][start_tidx:end_tidx]
                    
                    _, _, _, normalized_actions_mean, normalized_actions_logscale = get_vla_action_batch(
                        cfg,
                        vla=vla,
                        obs_batch=batch_obs_input,
                        task_label=task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        scale_head=scale_head,
                        use_film=cfg.use_film,
                        use_laplace_sampling=False,
                    )
                    batch_log_probs = laplace_log_prob(normalized_actions_mean.to(torch.bfloat16),  normalized_actions_logscale.to(torch.bfloat16), batch_ref_action_normalized.to(torch.bfloat16), cfg.log_scale_clip) # (batch_size, num_steps, action_dim)

                    if self.log_prob_mode == 'sum_on_action_dim':
                        batch_log_probs = batch_log_probs.sum(dim=-1) # (batch_size, num_steps, action_dim) -> (batch_size, num_steps)
                        batch_ref_logprob = batch_ref_logprob.sum(dim=-1)
                    elif self.log_prob_mode == 'avg_on_action_dim':
                        batch_log_probs = batch_log_probs.mean(dim=-1) # (batch_size, num_steps, action_dim) -> (batch_size, num_steps)
                        batch_ref_logprob = batch_ref_logprob.mean(dim=-1)
                    else:
                        pass

                    valid_value_count += batch_log_probs.shape[0] * batch_log_probs.shape[1]

                    ratio = (batch_log_probs - batch_ref_logprob).exp()

                    pg_losses = -all_advantage[bidx] * ratio
                    pg_losses2 = -all_advantage[bidx] * torch.clamp(
                        ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_high
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    loss = pg_loss_max.sum() / num_total_tokens / self.gradient_accumulation_steps


                    loss.backward()
                
                    with torch.no_grad():
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        pg_clipfrac_stats.append(pg_clipfrac.detach().item())
                        pg_loss_stats.append(loss.detach().item())
                        pg_ratio_stats.append(ratio.mean().detach().item())
                

                gradient_step += 1

                if gradient_step % self.gradient_accumulation_steps == 0:
                    # Synchronize gradients in distributed setting
                    if dist.is_initialized():
                        for param in model.trainable_params['model']:
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                        for param in model.trainable_params['header']:
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                    grad_norm_model = torch.nn.utils.clip_grad_norm_(model.trainable_params['model'], self.grad_norm_clip_model)
                    grad_norm_header = torch.nn.utils.clip_grad_norm_(model.trainable_params['header'], self.grad_norm_clip_header)
                    
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    gradient_step = 0
                    gradient_norm_model_stats.append(grad_norm_model.detach().item())
                    gradient_norm_header_stats.append(grad_norm_header.detach().item())

        # Free memory for all episodes
        del all_episodes
        gc.collect()

        metrics = {
            'pg_clipfrac_stats': torch.tensor(pg_clipfrac_stats).mean(),
            'pg_loss_stats': torch.tensor(pg_loss_stats).mean(),
            'pg_ratio_stats': torch.tensor(pg_ratio_stats).mean(),
            'mean_scores': torch.tensor(all_scores).mean(),
            'mean_advantage': all_advantage[valid_mask].mean(),
            'mean_rlhf_reward': rlhf_reward.mean(),
            'rollout_checked': torch.tensor(samples_checked),
            'gradient_norm_model_stats': torch.tensor(gradient_norm_model_stats).mean(),
            'gradient_norm_header_stats': torch.tensor(gradient_norm_header_stats).mean(),
            'non_zero_adv_ratio': (all_advantage[valid_mask] != 0).float().mean()
        }
                
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                if dist.is_initialized():
                    metrics[key] = metrics[key].to(model.device)
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
                task_srs[f'rl_train_succeess_rate/{task_name}'] = torch.tensor(all_scores).to(model.device)[valid_task_mask].mean().item()
        
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