import numpy as np
import gc
import ript.utils.libero_utils as lu
import ript.utils.obs_utils as ObsUtils
import wandb
from tqdm import tqdm
import multiprocessing

class LiberoRunner():
    def __init__(self,
                 env_factory,
                 benchmark_name,
                 mode, # all or few
                 rollouts_per_env,
                 num_parallel_envs,
                 max_episode_length,
                 frame_stack=1,
                 fps=10,
                 debug=False,
                 task_embedding_format='clip',
                 task_names_to_use=[],
                 reset_type='ori',
                 ):
        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = lu.get_benchmark(benchmark_name)()
        descriptions = [self.benchmark.get_task(i).language for i in range(self.benchmark.n_tasks)]
        task_embs = lu.get_task_embs(task_embedding_format, descriptions)
        self.benchmark.set_task_embs(task_embs)
        
        self.env_names = self.benchmark.get_task_names()


        if task_names_to_use is not None and len(task_names_to_use) > 0:
            self.env_names_to_run = [env_name for env_name in self.env_names if env_name in task_names_to_use]
        else:
            self.env_names_to_run = self.env_names
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.frame_stack = frame_stack
        if num_parallel_envs>1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
        self.max_episode_length = max_episode_length
        self.fps = fps
        self.reset_type = reset_type

        print('Eval Runner Reset Type:', self.reset_type)
        
    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None, run_env_names=None, render=False):
        if run_env_names is None:
            env_names = self.env_names_to_run
        else:
            env_names = run_env_names
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):
            print(f'running {env_name}')
            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=render)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)

                if i < n_video and render:
                    if save_video_fn is not None:
                        video_hwc = np.array(episode['render'])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, i)
                    else:
                        env_video.extend(episode['render'])
                    
            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)
            print(f'number of rollouts: {len(successes)}')
            if len(env_video) > 0:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)

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

        if len(videos) > 0:
            output['rollout_videos'] = {}
        for env_name in videos:
            output['rollout_videos'][env_name] = videos[env_name]
        
        return output

    def create_env(self, env_name):
        env_id = self.env_names.index(env_name)
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        env_fn = lambda: lu.LiberoFrameStack(self.env_factory(env_id, self.benchmark), self.frame_stack)
        env = lu.LiberoVectorWrapper(env_fn, self.num_parallel_envs)

        return env, env_id, env_num

    def run_policy_in_env(self, env_name, policy, all_init_states=None, render=False, created_env=None, random_init=False):
        import time
        start = time.time()
        if created_env is None:
            env, env_id, env_num = self.create_env(env_name)
        else:
            env, env_id, env_num = created_env
        end = time.time()

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
            for k in range(env_num):
                episode_k = {key: value[:,k] for key, value in episode.items()}
                yield success[k], total_reward[k], episode_k
        env._env.close()
        gc.collect()
        del env
    
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        if self.reset_type == 'fast':
            obs, info = env.reset(init_states=init_states_)
        elif self.reset_type == 'ori':
            obs, info = env.reset_ori(init_states=init_states_)
        else:
            raise ValueError(f'reset_type {self.reset_type} not supported')

        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id, task_emb: policy_object.get_action(obs, task_id, task_emb)
        
        success, total_reward = [False]*env_num, [0]*env_num

        episode = {key: [value[:,-1]] for key, value in obs.items()}
        episode['actions'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = self.env_names.index(env_name)
        task_emb = self.benchmark.get_task_emb(task_id).repeat(env_num, 1)
        steps = 0
        while steps < self.max_episode_length:
            action = policy(obs, task_id, task_emb)
            # action = env.action_space.sample()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs
            for key, value in obs.items():
                episode[key].append(value[:,-1])
            episode['actions'].append(action)
            if render:
                episode['render'].append(env.render())
        
            for k in range(env_num):
                success[k] = success[k] or terminated[k]
            
            if all(success):
                break
            steps += 1
        print(f'\t completed with {steps} steps, success: {success}')
        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode

class LiberoRunner_rl(LiberoRunner):
    '''
    Adapting from LiberoRunner to enable context caching used for RIPT-VLA.
    '''
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        print('start run_episode')
        
        if self.reset_type == 'fast':
            obs, info = env.reset(init_states=init_states_)
        else:
            obs, info = env.reset_ori(init_states=init_states_)

        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id, task_emb: policy_object.get_action(obs, task_id, task_emb)
        
        success, total_reward = [False]*env_num, [0]*env_num

        episode = {key: [value[:,-1]] for key, value in obs.items()}
        episode['actions'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = self.env_names.index(env_name)
        task_emb = self.benchmark.get_task_emb(task_id).repeat(env_num, 1)
        steps = 0

        policy_inference_steps = 0
        env_policy_inference_steps = [0] * env_num
        all_context_tokens = []
        all_action_indices = []

        while steps < self.max_episode_length:
            action, context_tokens, action_indices = policy(obs, task_id, task_emb)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs
            for key, value in obs.items():
                episode[key].append(value[:,-1])
            episode['actions'].append(action)
            if render:
                episode['render'].append(env.render())
            
            if context_tokens is not None and action_indices is not None:
                policy_inference_steps += 1
                all_context_tokens.append(context_tokens)
                all_action_indices.append(action_indices)
        
            for k in range(env_num):
                if not success[k] and terminated[k]:
                    env_policy_inference_steps[k] = policy_inference_steps
                success[k] = success[k] or terminated[k]
            
            if all(success):
                break
            steps += 1
        
        for k in range(env_num):
            if env_policy_inference_steps[k] == 0:
                env_policy_inference_steps[k] = policy_inference_steps

        print(f'\t completed with {steps} steps, success: {success}')
        episode = {key: np.array(value) for key, value in episode.items()}
        episode['context_tokens'] = np.stack(all_context_tokens, axis=0) # (T, K, L_obs, D)
        episode['action_indices'] = np.stack(all_action_indices, axis=0) # (T, K, L_act)
        episode['policy_inference_steps'] = np.array(env_policy_inference_steps).reshape(1, env_num) # (1, K)

        return success, total_reward, episode
