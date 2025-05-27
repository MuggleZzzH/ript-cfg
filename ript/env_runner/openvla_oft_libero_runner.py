import os
import logging
import numpy as np
import gc
import sys
from tqdm import tqdm
import math
from enum import Enum
import multiprocessing
from collections import deque
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

from PIL import Image
import tensorflow as tf
import torch
import time
import numpy as np

import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from experiments.robot.openvla_utils import (
    check_image_format,
    prepare_images_for_vla,
    normalize_proprio,
)

from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)

def laplace_log_prob(mu, log_b, action, log_scale_clip):
    """
    Computes log-prob of an action under a factorized Laplace distribution.

    Args:
        mu: Tensor of shape (batch_size, action_dim) - predicted mean
        log_b: Tensor of shape (batch_size, action_dim) - predicted log-scale
        action: Tensor of shape (batch_size, action_dim) - sampled action
        log_scale_clip: float - clip log-scale to this value
    Returns:
        log_prob: Tensor of shape (batch_size, action_dim) - log-probability for each sample per action dimension
    """

    if log_scale_clip is not None:
        log_b = torch.clamp(log_b, min=log_scale_clip[0], max=log_scale_clip[1])

    b = torch.exp(log_b)
    abs_error = torch.abs(action - mu)
    log_prob_per_dim = -log_b - abs_error / b - math.log(2.0)
    log_prob = log_prob_per_dim
    return log_prob

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay

def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

def get_vla_action_batch(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs_batch: List[Dict[str, Any]],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    scale_head: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    use_laplace_sampling: bool = False,
    scale_factor: float = 1.0,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    DEVICE = vla.device
    batch_size = len(obs_batch)

    # Collect all input images
    all_full_images = [obs["full_image"].copy() for obs in obs_batch]
    all_wrist_images = [obs["wrist_image"].copy() for obs in obs_batch]

    # Process images
    all_full_images = prepare_images_for_vla(all_full_images, cfg)
    all_wrist_images = prepare_images_for_vla(all_wrist_images, cfg)

    # Build VLA prompt
    prompts = [f"In: What action should the robot take to {task_label.lower()}?\nOut:" for _ in range(batch_size)]

    # Process primary image
    all_primary_inputs = processor(prompts, all_full_images).to(DEVICE, dtype=torch.bfloat16)

    # Process additional wrist images if any
    all_wrist_inputs = processor(prompts, all_wrist_images).to(DEVICE, dtype=torch.bfloat16)
    
    # Concatenate all images
    all_primary_inputs["pixel_values"] = torch.cat([all_primary_inputs["pixel_values"], all_wrist_inputs["pixel_values"]], dim=1)

    # Process proprioception data if used
    proprio = None
    if cfg.use_proprio:
        proprio_batch = []
        for obs in obs_batch:
            proprio = obs["state"].copy()
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio, proprio_norm_stats)
            proprio_batch.append(proprio)
        proprio = np.stack(proprio_batch, axis=0)
    

    actions_unnormalized, normalized_actions, log_prob, normalized_actions_mean, normalized_actions_logscale = predict_action_batch(
        vla,
        input_ids=all_primary_inputs['input_ids'],
        attention_mask=all_primary_inputs['attention_mask'],
        pixel_values=all_primary_inputs['pixel_values'],
        unnorm_key=cfg.unnorm_key,
        proprio=proprio,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        action_head=action_head,
        scale_head=scale_head,
        use_film=use_film,
        use_laplace_sampling=use_laplace_sampling,
        scale_factor=scale_factor,
        log_scale_clip=cfg.log_scale_clip
    )

    # Extract subset of actions for open loop steps
    return actions_unnormalized, normalized_actions, log_prob, normalized_actions_mean, normalized_actions_logscale


def predict_action_batch(
    vla: torch.nn.Module,
    input_ids: Optional[torch.LongTensor] = None,
    unnorm_key: Optional[str] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    scale_head=None,
    use_film: bool = False,
    use_laplace_sampling: bool = False,
    scale_factor: float = 1.0,
    log_scale_clip: Optional[Tuple[float, float]] = None,
    **kwargs: str,
) -> np.ndarray:
    """Predict actions from input sequence, with options for different prediction methods.

    Args:
        input_ids: Input token ids
        unnorm_key: Key for unnormalization statistics
        proprio: Proprioceptive features
        proprio_projector: Projector for proprioceptive features
        action_head: Optional head for L1 regression or diffusion-based prediction
        noisy_action_projector: Projector for noisy actions in diffusion-based prediction
        use_film: Whether to use FiLM conditioning
        **kwargs: Additional arguments including pixel_values and attention_mask

    Returns:
        Tuple of (unnormalized_actions, action_hidden_states)
    """
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    batch_size = input_ids.shape[0]
    
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).repeat(batch_size, 1).to(input_ids.device)), dim=1
        )

    pixel_values = kwargs["pixel_values"]
    attention_mask = kwargs["attention_mask"]

    # Create fake labels tensor (needed for action mask)
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    # Get number of tokens in prompt (excluding the start token)
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

    # Prepare inputs by adding necessary tokens
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)

    # Update labels tensor for action mask computation later
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

    # Get input embeddings and action masks
    input_embeddings = vla.get_input_embeddings()(input_ids)
    all_actions_mask = vla._process_action_masks(labels)

    # Process vision features
    with torch.no_grad(): # never need to backprop through vision features
        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        projected_patch_embeddings = vla._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = vla._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

    # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if use_proprio:
        NUM_PATCHES += 1
    if use_diffusion:
        NUM_PATCHES += 1

    # Run regression or discrete token-based prediction
    normalized_actions_mean, normalized_actions_logscale = _regression_or_discrete_prediction_batch(
        vla,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head,
        scale_head,
    )

    # sample during rollout collection, not during evaluation
    if use_laplace_sampling:
        action_b = torch.exp(normalized_actions_logscale) * scale_factor
        normalized_actions = torch.distributions.Laplace(loc=normalized_actions_mean, scale=action_b).sample()

    else:
        normalized_actions = normalized_actions_mean

    # Unnormalize predicted actions
    log_prob = laplace_log_prob(normalized_actions_mean, normalized_actions_logscale, normalized_actions, log_scale_clip)
    actions_unnormalized = vla._unnormalize_actions(normalized_actions.float().cpu().detach().numpy(), unnorm_key)

    return actions_unnormalized, normalized_actions, log_prob, normalized_actions_mean, normalized_actions_logscale

def _regression_or_discrete_prediction_batch(
    vla,
    input_embeddings,
    all_actions_mask,
    projected_patch_embeddings,
    attention_mask,
    labels,
    NUM_PATCHES,
    NUM_PROMPT_TOKENS,
    action_head=None,
    scale_head=None,
):
    """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
    # Zero out action token embeddings
    all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    input_embeddings = input_embeddings * ~all_actions_mask

    # Build multimodal embeddings and attention mask
    multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )


    # Step 1: Forward pass with prompt tokens (always no grad)
    with torch.no_grad():
        prompt_embeddings = multimodal_embeddings[:, :NUM_PROMPT_TOKENS + NUM_PATCHES, :]
        prompt_attention_mask = multimodal_attention_mask[:, :NUM_PROMPT_TOKENS + NUM_PATCHES]

        prompt_lm_output = vla.language_model(
            input_ids=None,
            attention_mask=prompt_attention_mask,
            inputs_embeds=prompt_embeddings,
            use_cache=True,
            return_dict=True,
        )
        kv_cache = prompt_lm_output.past_key_values

    # Step 2: Forward pass with action tokens (with grad)
    action_embeddings = multimodal_embeddings[:, NUM_PROMPT_TOKENS + NUM_PATCHES:, :]
    action_attention_mask = multimodal_attention_mask[:, NUM_PROMPT_TOKENS + NUM_PATCHES:]

    full_attention_mask = torch.cat([prompt_attention_mask, action_attention_mask], dim=1)

    prompt_len = NUM_PROMPT_TOKENS + NUM_PATCHES                         # 544
    new_len    = action_embeddings.shape[1]                               # 58

    # build position_ids for the new tokens
    position_ids = (
        torch.arange(prompt_len, prompt_len + new_len, device=action_embeddings.device)
        .unsqueeze(0)
        .expand(action_embeddings.shape[0], -1)
    )

    # Action forward pass
    # Note (Shuhan): we need to patch the forward method of LlamaSdpaAttention to handle the case where the input_embeddings has longer sequence length.
    # The current implementation of LlamaSdpaAttention does not support this; we monkey patch it in ript/algos/rl_optimizers/openvla_oft_interface.py
    action_lm_output = vla.language_model(
        input_ids=None,
        attention_mask=full_attention_mask,
        past_key_values=kv_cache,
        inputs_embeds=action_embeddings,
        use_cache=True,
        return_dict=True,
        position_ids=position_ids,
        output_hidden_states=True,
    )

    # Extract hidden states for action tokens
    last_hidden_states = action_lm_output.hidden_states[-1]  # (B, seq_len, D)
    actions_hidden_states = last_hidden_states[:, :-2, :]
    # print(actions_hidden_states.shape)
    
    # L1 regression prediction
    BATCH_SIZE = actions_hidden_states.shape[0]
    normalized_actions_mean = action_head.predict_action(actions_hidden_states)
    normalized_actions_mean = normalized_actions_mean.reshape(BATCH_SIZE, NUM_ACTIONS_CHUNK, ACTION_DIM)

    normalized_actions_logscale = scale_head(actions_hidden_states)
    normalized_actions_logscale = normalized_actions_logscale.reshape(BATCH_SIZE, NUM_ACTIONS_CHUNK, ACTION_DIM)

    return normalized_actions_mean, normalized_actions_logscale

class OpenVLAOFTLiberoRunner():
    def __init__(self,
                 benchmark_name,
                 rollouts_per_env,
                 num_parallel_envs,
                 max_episode_length=None,
                 task_names_to_use=[],
                 use_laplace_sampling=False,
                 scale_factor=1.0,
                 ):
        self.benchmark_name = benchmark_name.lower()

        benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark = benchmark_dict[benchmark_name.lower()]()
        self.env_names = self.benchmark.get_task_names()

        if task_names_to_use is not None and len(task_names_to_use) > 0:
            self.env_names_to_run = [env_name for env_name in self.env_names if env_name in task_names_to_use]
        else:
            self.env_names_to_run = self.env_names
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        if num_parallel_envs>1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
    
        if max_episode_length is None:
            self.max_episode_length = TASK_MAX_STEPS[self.benchmark_name]
        else:
            self.max_episode_length = max_episode_length
        
        self.num_steps_wait = 10
        self.use_laplace_sampling = use_laplace_sampling
        self.scale_factor = scale_factor

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None, run_env_names=None, created_envs=None):
        if run_env_names is None:
            env_names = self.env_names_to_run
        else:
            env_names = run_env_names
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        # videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):
            print(f'running {env_name}')
            if created_envs is None:
                created_env = None
            else:
                task_idx = self.env_names_to_run.index(env_name)
                created_env = created_envs[task_idx]
                print('using existing env')
            any_success = False
            env_succs, env_rews = [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0, created_env=created_env)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)

            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)
            print(f'number of rollouts: {len(successes)}')

        output = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(successes),
            'overall_average_reward': np.mean(rewards),
            'environments_solved': int(np.sum(per_env_any_success)),
            'rollout_count': len(successes),
        }
        output['rollout_success_rate'] = {}
        for env_name in env_names:
            output['rollout_success_rate'][env_name] = per_env_success_rates[env_name]
        return output

    def create_env(self, env_name):
        task_id = self.env_names.index(env_name)
        task = self.benchmark.get_task(task_id)
        
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        def get_env(task):
            env, _ = get_libero_env(task, 'openvla', resolution=256)
            return env
        env_factory = lambda: get_env(task)
        env = SubprocVectorEnv([env_factory for _ in range(env_num)])

        return env, task_id, env_num

    def run_policy_in_env(self, env_name, policy, all_init_states=None, render=False, created_env=None, random_init=False):
        import time
        start = time.time()
        if created_env is None:
            env, env_id, env_num = self.create_env(env_name)
        else:
            env, env_id, env_num = created_env
        end = time.time()
        # print('time taken for env_fn', end - start)

        if all_init_states is None:
            all_init_states = self.benchmark.get_task_init_states(env_id)
        count = 0
        eval_loop_num = (self.rollouts_per_env+self.num_parallel_envs-1)//self.num_parallel_envs

        print(f'eval_loop_num: {eval_loop_num}')

        while count < eval_loop_num:
            indices = np.arange(count * env_num, (count + 1) * env_num) % all_init_states.shape[0]
            print(f'indices: {indices}')
            if random_init:
                init_states_ = None
            else:
                init_states_ = all_init_states[indices]
            import time
            start = time.time()
            print('start eval loop')
            success, total_reward, episode = self.run_episode(env, 
                                                              env_name, 
                                                              policy,
                                                              init_states_,
                                                              env_num,
                                                              render)
            end = time.time()
            count += 1
            step_num = len(episode['actions'])
            for k in range(env_num):
                episode_k = {}
                for key in episode.keys():
                    episode_k[key] = [episode[key][i][k] for i in range(step_num)]
                yield success[k], total_reward[k], episode_k
        
        if created_env is None:
            env.close()
            gc.collect()
            del env
    
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        cfg = policy.cfg
        
        processor = policy.processor
        if hasattr(policy.model, 'module'):
            model = policy.model.module
        else:
            model = policy.model
        
        if hasattr(policy.action_head, 'module'):
            action_head = policy.action_head.module
        else:
            action_head = policy.action_head
        
        if hasattr(policy.scale_head, 'module'):
            scale_head = policy.scale_head.module
        else:
            scale_head = policy.scale_head
        
        if hasattr(policy.proprio_projector, 'module'):
            proprio_projector = policy.proprio_projector.module
        else:
            proprio_projector = policy.proprio_projector
        
        noisy_action_projector = None

        print('resetting env')

        env.reset()
        env.set_init_state(init_states_)

        task_id = self.env_names.index(env_name)
        task = self.benchmark.get_task(task_id)
        task_description = task.language

        t = 0
        success = [False] * env_num
        done = [False] * env_num
        total_reward = [0] * env_num

        success = [False] * env_num
        episode = {}
        episode['actions'] = []
        episode['valid'] = []
        episode['actions_normalized'] = []
        episode['log_prob'] = []
        episode['observations'] = []
        episode['task_description'] = []

        batch_action_squeue = [deque(maxlen=8) for _ in range(env_num)]


        model_family = 'openvla'
        resize_size = 224

        print('starting episode')

        while t < self.max_episode_length + self.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < self.num_steps_wait:
                obs, _, done, _ = env.step([get_libero_dummy_action(model_family) for _ in range(env_num)])
                t += 1
                continue

            step_actions = [get_libero_dummy_action(model_family) for _ in range(env_num)]
            step_valid = [False] * env_num
            step_log_prob = [None] * env_num
            step_action_normalized = [None] * env_num
            step_observations = [None] * env_num
            
            step_input_obs_ids = []
            step_input_obs_list = []

            for bidx in range(env_num):
                if not success[bidx]:
                    step_valid[bidx] = True
                    step_input_obs_ids.append(bidx)
                    observation, _ = prepare_observation(obs[bidx], resize_size)
                    step_observations[bidx] = observation
                    step_input_obs_list.append(observation)

            conduct_inference = all(len(batch_action_squeue[i]) == 0 for i in step_input_obs_ids)
            if conduct_inference:
                # start = time.time()
                # if self.use_laplace_sampling:
                #     print('using laplace sampling')
                # else:
                #     print('do not use sampling!')
                with torch.inference_mode():
                    actions, actions_normalized, log_prob, _, _ = get_vla_action_batch(
                        cfg,
                        vla=model,
                        obs_batch=step_input_obs_list,
                        task_label=task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        scale_head=scale_head,
                        use_film=cfg.use_film,
                        use_laplace_sampling=self.use_laplace_sampling,
                        scale_factor=self.scale_factor
                    )
                    # end = time.time()
                    # print(f"Batch Size: {len(step_input_obs_list)}, Time taken: {end - start} seconds")
                    # print(log_prob.mean())
                    
                    for act_idx, obs_id in enumerate(step_input_obs_ids):
                        batch_action_squeue[obs_id].extend(actions[act_idx])

            for i in range(len(step_input_obs_list)):
                bidx = step_input_obs_ids[i]
                action = batch_action_squeue[bidx].popleft()
                action = process_action(action, 'openvla')
                step_actions[bidx] = action.tolist()
                step_action_normalized[bidx] = actions_normalized[i].float().detach().cpu().numpy()
                step_log_prob[bidx] = log_prob[i].float().detach().cpu().numpy()

            obs, reward, done, info = env.step(step_actions)
            # print(done)

            for k in range(env_num):
                success[k] = success[k] or done[k]

            if all(success):
                break

            t += 1

            # if t % 32 == 0:
            #     print('\t\tt', t, 'success', success)
            
            if conduct_inference:
                episode['actions'].append(step_actions)
                episode['valid'].append(step_valid)
                episode['actions_normalized'].append(step_action_normalized)
                episode['log_prob'].append(step_log_prob)
                episode['observations'].append(step_observations)
                episode['task_description'].append([task_description for _ in range(len(step_actions))])
        print('\tsuccess', success)
        print('\t\tsaved episodes:', len(episode['actions']))
        return success, total_reward, episode