from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import gc
import os
from tqdm import tqdm
import json
from ript.algos.rl_optimizers.file_counter import (
    setup_file_counter, 
    reset_global_counter, 
    cleanup_counter
)
from .model_interface import RLModelInterface
import hashlib

def compute_hash_from_state(state, bidx):
  state_data = state['states'][bidx][0]
  state_mask = state['pad_mask'][bidx]
  return hashlib.sha256(state_data[state_mask].cpu().numpy().tobytes()).hexdigest()

# Set multiprocessing start method to 'spawn' instead of 'fork' to prevent memory copying
import multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Failed to set multiprocessing start method to 'spawn', it may already be set")


class RolloutGenerator:
    """
    Generates rollouts for reinforcement learning algorithms.
    Manages the interaction between the model and environment to collect experience.
    """

    def __init__(
        self,
        rloo_batch_size: int = 8,
        demo_batch_size: int = 1,
        use_tqdm: bool = True,
        early_stop_percentage: float = 1.0,
        enable_dynamic_sampling: bool = False,
        task_names_to_use: Optional[List[str]] = None,
        env_runner: Optional[Any] = None,
        create_env: bool = True,
        use_val_init: bool = False,
        mix_val_init_in_rloo: bool = False,
        rollout_stats_path: Optional[str] = None,
        enable_rollout_stats_tracking: bool = False,
    ):
        """
        Initialize the rollout generator.
        
        Args:
            model_adapter: The model adapter for action prediction
            reward_func: Function that evaluates actions and returns rewards/metrics
            max_steps: Maximum number of steps per rollout
            rloo_batch_size: Number of rollouts to generate
            demo_batch_size: Batch size for parallel rollout generation
            use_tqdm: Whether to use tqdm for progress tracking
            early_stop_percentage: Percentage of batch size to reach before early stopping
            enable_dynamic_sampling: Whether to enable dynamic sampling
            use_full_action_tokens: Whether to use full action tokens
            task_names_to_use: List of task names to use
            env_runner: Environment runner for executing rollouts
            create_env: Whether to create environments during initialization
            use_val_init: Whether to use validation environment for initialization
            mix_val_init_in_rloo: Whether to randomly sample multiple validation initializations in each rloo batch
        """
        self.rloo_batch_size = rloo_batch_size
        self.demo_batch_size = demo_batch_size
        self.use_tqdm = use_tqdm
        self.early_stop_percentage = early_stop_percentage
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.task_names_to_use = task_names_to_use or []
        self.env_runner = env_runner
        self.created_envs = []
        
        # Setup file counter for distributed coordination
        self.file_counter, self.counter_filename = setup_file_counter()
        
        # Calculate global rollout demo threshold
        if dist.is_initialized():
            world_size = dist.get_world_size()
            total_demo_demo_batch_size = self.demo_batch_size * world_size
            self.global_rollout_demo_threshold = int(total_demo_demo_batch_size * self.early_stop_percentage)
        else:
            self.global_rollout_demo_threshold = int(self.demo_batch_size * self.early_stop_percentage)
        
        # Create environments if requested
        if create_env and self.env_runner is not None and self.task_names_to_use:
            self._create_environments()
        
        self.use_val_init = use_val_init
        self.mix_val_init_in_rloo = mix_val_init_in_rloo

        self.rollout_stats = {} # key: init_hash, value: list of success
        self.rollout_skip_cnt = {} # key: init_hash, value: number of rounds skipped
        self.rollout_skip_threshold = 3
        self.enable_rollout_stats_tracking = enable_rollout_stats_tracking
        if rollout_stats_path is not None:
            if '*' in rollout_stats_path:
                import glob
                rollout_stats_paths = glob.glob(rollout_stats_path)
                for path in rollout_stats_paths:
                    rollout_stats = json.load(open(path, 'r'))
                    self.rollout_stats.update(rollout_stats)
                    print(f"Loaded rollout stats from {path}")
            elif os.path.exists(rollout_stats_path):
                rollout_stats = json.load(open(rollout_stats_path, 'r'))
                self.rollout_stats = rollout_stats
                print(f"Loaded rollout stats from {rollout_stats_path}")
            else:
                print(f"Rollout stats path {rollout_stats_path} does not exist, start tracking from scratch")
            for init_hash in self.rollout_stats:
                self.rollout_skip_cnt[init_hash] = 0
        else:
            print(f"No rollout stats path provided, start tracking from scratch")

    def _create_environments(self):
        """Create environments for each task"""
        if self.use_tqdm:
            task_iterator = tqdm(self.task_names_to_use, desc='Creating environments')
        else:
            task_iterator = self.task_names_to_use
        
        self.created_envs = []
        for task_name in task_iterator:
            self.created_envs.append(self.env_runner.create_env(task_name))
    
    def _close_environments(self):
        """Close environments"""
        for created_env in self.created_envs:
            env, env_id, env_num = created_env
            env.close()
            gc.collect()
            del env
            print(f"Closed environment {env_id}")
        torch.cuda.empty_cache()
        del self.created_envs

    def generate_rollouts(self, model, batch, data_iterator, dataloader):
        """
        Generate rollout episodes for RL optimization.
        
        Args:
            model: The model to use for generating rollouts
            batch: The batch of data to use for rollouts
            data_iterator: Iterator for getting more batches if needed
            dataloader: Dataloader for reinitializing the iterator if exhausted
            
        Returns:
            tuple: (episodes, task_ids, valid_mask, samples_checked)
        """
        # Reset the global counter
        reset_global_counter(self.file_counter)
        
        all_successes = []
        all_scores = []
        all_episodes = []
        all_task_ids = []
        
        # Original batch size (number of valid samples we want)
        demo_batch_size = batch['task_id'].shape[0]
        valid_samples = 0
        
        # Use a pointer into the current batch; if it is exhausted, fetch a new batch
        current_batch = batch
        batch_index = 0
        samples_checked = 0
        
        early_stop = False
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        while valid_samples < demo_batch_size and not early_stop:
            samples_checked += 1
            
            # If current batch is used up, get a new batch from the data iterator
            if batch_index >= current_batch['task_id'].shape[0]:
                try:
                    current_batch = next(data_iterator)
                    batch_index = 0
                except StopIteration:
                    # Reinitialize the data iterator using the dataloader
                    data_iterator = iter(dataloader)
                    current_batch = next(data_iterator)
                    batch_index = 0
            
            # Process one sample from the current batch
            sample_task_id = current_batch['task_id'][batch_index].item()
            task_idx = sample_task_id
            task_name = self.task_names_to_use[task_idx]
            created_env = self.created_envs[task_idx]
            
            # Get environment initialization
            if self.use_val_init:
                # Use random initialization from the testing set
                _, env_id, env_num = created_env
                all_env_init_states = self.env_runner.benchmark.get_task_init_states(env_id)
                all_env_num = len(all_env_init_states)
                if self.mix_val_init_in_rloo:
                    select_idx = np.random.randint(0, all_env_num, size=self.rloo_batch_size)
                    env_init_states = all_env_init_states[select_idx]
                    print(f"Rank {rank} using {select_idx} validation initializations for RLOO")
                else:
                    select_idx = np.random.randint(0, all_env_num)
                    # select_idx = 4
                    env_init_state = all_env_init_states[select_idx]
                    env_init_states = np.tile(env_init_state, (self.rloo_batch_size, 1))
                    print(f"Rank {rank} using {select_idx} validation initialization for RLOO")
                random_init = False
            else:
                # Extract init_state for this sample
                sample_states = current_batch['init_state']
                init_state = sample_states['states'][batch_index, 0][sample_states['pad_mask'][batch_index]]
                env_init_states = init_state.unsqueeze(0).repeat(self.rloo_batch_size, 1).cpu().numpy()
                random_init = False
            
            init_hash = compute_hash_from_state(current_batch['init_state'], batch_index)
            
            # track the rollout stats and skip all-successful initializations previously seen
            if self.enable_rollout_stats_tracking and init_hash in self.rollout_stats:
                new_rollout_successes = self.rollout_stats[init_hash][-self.rloo_batch_size:]
                if all(s == 1 for s in new_rollout_successes):
                    print(f"Rank {rank} skipping sample at batch index {batch_index} for task {task_name} "
                         f"due to new rollouts (hash: {init_hash}) being all successful")
                    batch_index += 1
                    self.rollout_skip_cnt[init_hash] += 1
                    if self.rollout_skip_cnt[init_hash] > self.rollout_skip_threshold:
                        print(f"Rank {rank} removing init_hash: {init_hash} from rollout_stats because it has been skipped {self.rollout_skip_cnt[init_hash]} times")
                        del self.rollout_stats[init_hash]
                    continue
            else:
                self.rollout_stats[init_hash] = []
                self.rollout_skip_cnt[init_hash] = 0

            print(f'Rank {rank} running rollouts for init_hash: {init_hash}, rollout_stats: {self.rollout_stats[init_hash]}')

            # Run rollouts for this sample
            rollout_env = self.env_runner.run_policy_in_env(
                task_name, 
                model,
                env_init_states,
                render=False, 
                created_env=created_env,
                random_init=random_init
            )
            sample_successes = []
            sample_scores = []
            sample_episodes = []
            sample_task_ids = []
            
            for _ in range(self.rloo_batch_size):
                try:
                    success, total_reward, episode = next(rollout_env)
                except StopIteration:
                    break
                episode['init_hash'] = init_hash
                sample_successes.append(success)
                sample_scores.append(total_reward)
                sample_episodes.append(episode)
                sample_task_ids.append(task_idx)
                self.rollout_stats[init_hash].append(int(success))

            # Check if the sample's rollouts are all 0 or all 1
            if (self.enable_dynamic_sampling and 
                len(sample_successes) > 0 and 
                (all(s == 0 for s in sample_successes) or all(s == 1 for s in sample_successes))):
                # Discard this sample
                print(f"Rank {rank} discarding sample at batch index {batch_index} for task {task_name} "
                     f"due to uniform rollout successes: {sample_successes}")
            else:
                # Accept the sample
                all_successes.extend(sample_successes)
                all_scores.extend(sample_scores)
                all_episodes.extend(sample_episodes)
                all_task_ids.extend(sample_task_ids)
                valid_samples += 1
                
                # Update the global counter
                self.file_counter.update(1)
            
            current_global = self.file_counter.get()
            if rank == 0:  # Only print from rank 0 to avoid cluttering the output
                print(f"Global rollout demo counter {current_global}/{self.global_rollout_demo_threshold}")
            if current_global >= self.global_rollout_demo_threshold:
                early_stop = True
            
            batch_index += 1
        
        # Padding for missing rollouts
        target_rollouts = self.demo_batch_size * self.rloo_batch_size
        valid_mask = [True] * len(all_successes)
        if len(all_successes) < target_rollouts:
            num_pad = target_rollouts - len(all_successes)
            if len(all_successes) > 0:
                last_success = all_successes[-1]
                last_score = all_scores[-1]
                last_episode = all_episodes[-1]
                last_task_id = all_task_ids[-1]
            else:
                last_success = False
                last_score = 0.0
                last_episode = {
                    "context_tokens": [0],
                    "action_indices": [0],
                    "policy_inference_steps": [1]
                }
                last_task_id = 0
            
            for _ in range(num_pad):
                all_successes.append(last_success)
                all_scores.append(last_score)
                all_episodes.append(last_episode)
                all_task_ids.append(last_task_id)
                valid_mask.append(False)
        
        # Process the rollout data into tensors
        device = model.device
        
        # Add success field to episodes
        for i, success in enumerate(all_successes):
            all_episodes[i]['success'] = success
        
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool).to(device)
        
        if dist.is_initialized():
            dist.barrier()
        

        print(f"Rank {rank} closing environments")

        return (
            all_episodes, 
            torch.tensor(all_task_ids).to(device), 
            valid_mask,
            samples_checked
        )
    
    def cleanup(self):
        """Clean up the rollout generator resources"""
        cleanup_counter(self.counter_filename)
        self._close_environments()