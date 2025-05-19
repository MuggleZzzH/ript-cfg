import numpy as np

import ript.utils.metaworld_utils as mu
import wandb
from tqdm import tqdm


class MetaWorldRunner():
    def __init__(self,
                 env_factory,
                 benchmark_name,
                 mode, # train or test
                 rollouts_per_env,
                 fps=10,
                 debug=False,
                 random_task=False,
                 num_parallel_envs=None,
                 reset_type='ori'
                 ):
        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = mu.get_benchmark(benchmark_name) if not debug else None
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.fps = fps
        self.random_task = random_task
        
        self.env_names = mu.get_env_names(self.benchmark_name, self.mode)
        

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None, run_env_names=None):
        # print

        if run_env_names is None:
            env_names = self.env_names
        else:
            env_names = run_env_names

        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)

                # if i < n_video:
                #     if save_video_fn is not None:
                #         video_hwc = np.array(episode['render'])
                #         video_chw = video_hwc.transpose((0, 3, 1, 2))
                #         save_video_fn(video_chw, env_name, i)
                #     else:
                #         env_video.extend(episode['render'])
                    
            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)

            # if len(env_video) > 0:
            #     video_hwc = np.array(env_video)
            #     video_chw = video_hwc.transpose((0, 3, 1, 2))
            #     videos[env_name] = wandb.Video(video_chw, fps=self.fps)
            
        # output['rollout'] = {}
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
            # This metric isn't that useful
            # output[f'rollout_detail/average_reward_{env_name}'] = per_env_rewards[env_name]
        # if len(videos) > 0:
        #     output['rollout_videos'] = {}
        # for env_name in videos:

        #     output['rollout_videos'][env_name] = videos[env_name]
        
        return output

    def create_env(self, env_name):
        env_id = self.env_names.index(env_name)
        env = self.env_factory(env_name=env_name)
        tasks = mu.get_tasks(self.benchmark, self.mode)
        
        env_tasks = [task for task in tasks if task.env_name == env_name]
        return env, env_id, env_tasks

    def run_policy_in_env(self, env_name, policy, render=False, created_env=None, task_ind=None):
        env, _, env_tasks = self.create_env(env_name)
        # if created_env is None:
            # env, _, env_tasks = self.create_env(env_name)
        # else:
            # env, _, env_tasks = created_env
        count = 0
        # env.reset()
        with tqdm(total=self.rollouts_per_env, desc=f"Rollouts for {env_name}") as pbar:
            while count < self.rollouts_per_env:
                if len(env_tasks) > 0:
                    if task_ind is None:
                        task = env_tasks[count % len(env_tasks)]
                        print(f'running task {count % len(env_tasks)} of {env_name}')
                    else:
                        task = env_tasks[task_ind]
                        print(f'running task {task_ind} of {env_name}')
                    env.set_task(task)
                success, total_reward, episode = self.run_episode(env, 
                                                                env_name, 
                                                                policy,
                                                                render)
                count += 1
                # print(f"count+1: {count}")
                pbar.update(1)
                yield success, total_reward, episode
        
        env.close()
        del env

    def run_episode(self, env, env_name, policy, render=False):
        obs, _ = env.reset()
        # breakpoint()
        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            # breakpoint()
            policy = lambda obs, task_id: policy_object.get_action(obs, task_id)
        
        done, success, total_reward = False, False, 0

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode['actions'] = []
        episode['terminated'] = []
        episode['truncated'] = []
        episode['reward'] = []
        episode['success'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = mu.get_index(env_name)

        # count = 0

        while not done:
            obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            action = policy(obs, task_id).squeeze()
            # action = env.action_space.sample()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value[-1])
            episode['actions'].append(action)
            episode['terminated'].append(terminated)
            episode['truncated'].append(truncated)
            episode['reward'].append(reward)
            episode['success'].append(info['success'])
            if int(info["success"]) == 1:
                success = True
            if render:
                episode['render'].append(env.render())

            # count += 1
            # if count > 50:
            #     break

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode
    

class MetaWorldRunner_rl(MetaWorldRunner):
    def run_episode(self, env, env_name, policy, render=False):
        obs, _ = env.reset()
        # breakpoint()
        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            # breakpoint()
            policy = lambda obs, task_id: policy_object.get_action(obs, task_id)
        
        done, success, total_reward = False, False, 0
        policy_inference_steps = 0

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode['actions'] = []
        episode['terminated'] = []
        episode['truncated'] = []
        episode['reward'] = []
        episode['success'] = []
        all_context_tokens = []
        all_action_indices = []
        if render:
            episode['render'] = [env.render()]

        task_id = mu.get_index(env_name)

        count = 0


        while not done:
            obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            # action = policy(obs, task_id).squeeze()
            # action = env.action_space.sample()
            action, context_tokens, action_indices = policy(obs, task_id)
            action = np.clip(action, env.action_space.low, env.action_space.high).squeeze()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            if context_tokens is not None and action_indices is not None:
                policy_inference_steps += 1
                all_context_tokens.append(context_tokens)
                all_action_indices.append(action_indices)

            for key, value in obs.items():
                episode[key].append(value[-1])
            episode['actions'].append(action)
            episode['terminated'].append(terminated)
            episode['truncated'].append(truncated)
            episode['reward'].append(reward)
            episode['success'].append(info['success'])
            if int(info["success"]) == 1:
                success = True
            if render:
                episode['render'].append(env.render())

            count += 1
            # if count > 50:
            #     break
        episode = {key: np.array(value) for key, value in episode.items()}

        print(success)

        episode['context_tokens'] = np.stack(all_context_tokens, axis=0)[:, 0]
        episode['action_indices'] = np.stack(all_action_indices, axis=0)[:, 0]
        episode['policy_inference_steps'] = np.array(policy_inference_steps).reshape(1)
        
        return success, total_reward, episode
