import os
import logging
import numpy as np
import gc
import sys
from tqdm import tqdm
from enum import Enum
import multiprocessing
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)
from openvla.experiments.robot.openvla_utils import get_vla, crop_and_resize
from openvla.experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

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

def compute_log_probs(model, log_probs_ids, prompt_key_values):
  past_key_values = []
  for layer in range(len(prompt_key_values)):
    key_features = prompt_key_values[layer][0].to(model.device)
    value_features = prompt_key_values[layer][1].to(model.device)
    past_key_values.append((key_features, value_features))

  if hasattr(model, 'module'):
    logits = model.module.language_model(input_ids=log_probs_ids[:, :-1].to(model.device), past_key_values=past_key_values, use_cache=True)
  else:
    logits = model.language_model(input_ids=log_probs_ids[:, :-1].to(model.device), past_key_values=past_key_values, use_cache=True)
  logits = logits.logits

  logits = torch.nn.functional.log_softmax(logits, dim=-1)
  log_probs = torch.gather(logits, dim=-1, index=log_probs_ids[:, 1:, None].to(model.device))

  return log_probs


def top_k_sampling(logits, k, temperature):
  # Apply temperature scaling
  scaled_logits = logits / temperature
  # Find the top k values and indices
  top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
  # Compute probabilities from top values
  top_probs = torch.softmax(top_values, dim=-1)
  # Sample token index from the filtered probabilities
  sampled_indices = torch.multinomial(top_probs, num_samples=1, replacement=True)
  # Map the sampled index back to the original logits tensor
  original_indices = top_indices.gather(-1, sampled_indices)
  return original_indices


def vla_batch_generate(model, inputs, unnorm_key, beam_size, temperature):
  with torch.no_grad():
    model.eval()
    
    input_ids = inputs['input_ids']
    if not torch.all(input_ids[:, -1] == 29871):
      input_ids = torch.cat(
          (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).repeat(input_ids.shape[0], 1).to(input_ids.device)), dim=1
      )

    pixel_values = inputs['pixel_values']

    max_new_tokens = model.get_action_dim(unnorm_key)
    prompt_key_values = []
    # step_log_probs = []

    first_out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,  # <- only here
        use_cache=True,
    )
    past_key_values = first_out.past_key_values

    # make a clone of the past_key_values with detach, clone and cpu
    layers = len(past_key_values)
    for layer in range(layers):
      key_features = past_key_values[layer][0].detach().clone()[:, :, :-1]
      value_features = past_key_values[layer][1].detach().clone()[:, :, :-1] # not use the last token feature, because it is the input token for logits
      prompt_key_values.append((key_features, value_features))

    logits = first_out.logits[:, -1, :]
    next_token = top_k_sampling(logits, beam_size, temperature)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # step_log_probs.append(torch.gather(log_probs, dim=-1, index=next_token).squeeze(-1).detach().cpu())
    generated = [next_token]

    for _ in range(max_new_tokens-1):  # or however many new tokens you want
      out = model.language_model(
          input_ids=next_token,
          past_key_values=past_key_values,
          use_cache=True,
      )
      logits = out.logits[:, -1, :]
      # next_token = torch.argmax(logits, dim=-1, keepdim=True)
      next_token = top_k_sampling(logits, beam_size, temperature)
      generated.append(next_token)
      past_key_values = out.past_key_values
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
      # step_log_probs.append(torch.gather(log_probs, dim=-1, index=next_token).squeeze(-1).detach().cpu())


    predicted_action_token_ids = torch.concat(generated, dim=1).cpu()
    discretized_actions = model.vocab_size - predicted_action_token_ids.numpy()
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    # Unnormalize actions
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    # TODO: this is to it consistent with the log_prob computation in later stage
    log_probs_ids = predicted_action_token_ids
    log_probs_ids = torch.cat([torch.tensor([29871]).long()[None, ].repeat(log_probs_ids.shape[0], 1), log_probs_ids], dim=1)
    log_probs = compute_log_probs(model, log_probs_ids, prompt_key_values).detach().cpu()
    
    return actions, log_probs, log_probs_ids, prompt_key_values
  
def get_vla_action_with_info_batch(model, processor, batch_obs_list, task_label, unnorm_key, center_crop, beam_size, temperature):
  """Generates an action with the VLA policy."""
  
  input_images = []
  input_prompts = []
  for obs_i in batch_obs_list:
    image = Image.fromarray(obs_i)
    image = image.convert("RGB")
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    input_images.append(image)
    input_prompts.append(f"In: What action should the robot take to {task_label.lower()}?\nOut:")

  inputs = processor(input_prompts, input_images).to(model.device, dtype=torch.bfloat16)

  # print('batch size', len(input_images))

  # Get action.
  with torch.no_grad():
    actions, log_probs, log_probs_ids, prompt_key_values = vla_batch_generate(model, inputs, unnorm_key, beam_size, temperature)

  # rearrange output into list for each input observation
  batch_action_list = [actions[i] for i in range(len(actions))]
  batch_log_prob_list = [log_probs[i] for i in range(len(actions))]
  batch_log_probs_ids_list = [log_probs_ids[i] for i in range(len(actions))]
  batch_prompt_key_values_list = []
  for bidx in range(len(actions)):
    item_prompt_key_values = []
    for layer_id in range(len(prompt_key_values)):
      key_features = prompt_key_values[layer_id][0][bidx:bidx+1].cpu()
      value_features = prompt_key_values[layer_id][1][bidx:bidx+1].cpu()
      item_prompt_key_values.append((key_features, value_features))
    batch_prompt_key_values_list.append(item_prompt_key_values)
  
  return batch_action_list, batch_log_prob_list, batch_log_probs_ids_list, batch_prompt_key_values_list


class OpenVLALiberoRunner():
    def __init__(self,
                 benchmark_name,
                 rollouts_per_env,
                 num_parallel_envs,
                 max_episode_length=None,
                 task_names_to_use=[],
                 beam_size=5,
                 temperature=1.2,
                 episode_save_step=-1,
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
        self.beam_size = beam_size
        self.temperature = temperature
        self.episode_save_step = episode_save_step

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None, run_env_names=None):
        if run_env_names is None:
            env_names = self.env_names_to_run
        else:
            env_names = run_env_names
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        # videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):
            print(f'running {env_name}')
            any_success = False
            env_succs, env_rews = [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0)
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
        # env.close()
        # gc.collect()
        # del env
    
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        enable_episode_saving = self.episode_save_step > 0

        if hasattr(policy.model, 'module'):
            model = policy.model.module
        else:
            model = policy.model
        processor = policy.processor
        unnorm_key = policy.unnorm_key

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
        episode['logprob'] = []
        episode['logprob_ids'] = []
        episode['prompt_key_values'] = []

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
            step_log_prob_ids = [None] * env_num
            step_prompt_key_values = [None] * env_num
            
            step_input_obs_ids = []
            step_input_obs_list = []

            for bidx in range(env_num):
                if not success[bidx]:
                    step_valid[bidx] = True
                    step_input_obs_ids.append(bidx)
                    step_input_obs_list.append(get_libero_image(obs[bidx], resize_size))
                

            # start = time.time()
            batch_action_list, batch_log_prob_list, batch_log_probs_ids_list, batch_prompt_key_values_list = get_vla_action_with_info_batch(model, processor, step_input_obs_list, task_description, unnorm_key, center_crop=True, beam_size=self.beam_size, temperature=self.temperature)
            # end = time.time()
            # print(f"Batch Size: {len(step_input_obs_list)}, Time taken: {end - start} seconds")

            for i in range(len(step_input_obs_list)):
                bidx = step_input_obs_ids[i]
                
                action = batch_action_list[i]
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                step_actions[bidx] = action.tolist()
                
                step_log_prob[bidx] = batch_log_prob_list[i]
                step_log_prob_ids[bidx] = batch_log_probs_ids_list[i]
                step_prompt_key_values[bidx] = batch_prompt_key_values_list[i]

            obs, _, done, _ = env.step(step_actions)

            hit_success = False
            for k in range(env_num):
                if not success[k] and done[k]:
                    hit_success = True
                success[k] = success[k] or done[k]

            is_routine_saving = t % self.episode_save_step == 0
            is_last_step = t == self.max_episode_length - 1
            is_success_saving = hit_success

            if enable_episode_saving and (is_routine_saving or is_last_step or is_success_saving):
                print(f'\t\tsaving episode at step {t}')
                if is_success_saving:
                    print('\t\t\thit success')
                if is_routine_saving:
                    print('\t\t\tis routine saving')
                if is_last_step:
                    print('\t\t\tis last step')
                episode['actions'].append(step_actions)
                episode['valid'].append(step_valid)
                episode['logprob'].append(step_log_prob)
                episode['logprob_ids'].append(step_log_prob_ids)
                episode['prompt_key_values'].append(step_prompt_key_values)
            else:
                del step_actions
                del step_log_prob
                del step_log_prob_ids
                del step_prompt_key_values
                gc.collect()
        
            if all(success):
                break
            
            t += 1
            if t % 5 == 0:
                print('\t\tt', t, 'success', success)

        print('\tsuccess', success)
        print('\t\tsaved episodes:', len(episode['actions']))
        return success, total_reward, episode