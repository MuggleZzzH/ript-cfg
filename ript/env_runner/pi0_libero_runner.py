"""
PI0 Libero runner with open-loop=50 execution and episode fields aligned with rollout_generator expectations.
真实环境版（轻量）：创建 SubprocVectorEnv、使用 init_states、wait steps，然后以 50 步 chunk 执行。
"""

from typing import Any, Dict, Iterable, List, Tuple
import os
import numpy as np
import math
from collections import deque

from libero.libero import benchmark as libero_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from .pi0_runner_utils import build_pi0_observation, denorm_action_sequence, extract_state_8d


class Pi0LiberoRunner:
    def __init__(
        self,
        benchmark_name: str = "libero_spatial",
        rollouts_per_env: int = 1,
        num_parallel_envs: int = 1,
        max_episode_length: int | None = None,
        task_names_to_use: List[str] | None = None,
        num_steps_wait: int = 10,  # 额外等待步数
        **kwargs: Any,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.max_episode_length = max_episode_length or 300
        self.task_names_to_use = task_names_to_use or []
        self.num_steps_wait = num_steps_wait

        # Fallback stats; real values应由 policy.norm_stats 注入runner调用处
        self.state_mean = np.zeros(8, dtype=np.float32)
        self.state_std = np.ones(8, dtype=np.float32)
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)

    def run(
        self,
        policy,
        n_video: int = 0,
        do_tqdm: bool = False,
        save_video_fn=None,
        run_env_names: List[str] | None = None,
        created_envs: Any = None,
    ) -> Dict[str, Any]:
        try:
            from tqdm import tqdm  # optional
        except Exception:
            def tqdm(x, **kwargs):
                return x

        env_names = run_env_names if run_env_names is not None else (self.task_names_to_use or [self.benchmark_name])

        successes_all: List[float] = []
        rewards_all: List[float] = []
        per_env_success_rates: Dict[str, float] = {}
        per_env_any_success: List[bool] = []

        for env_name in tqdm(env_names, disable=not do_tqdm):
            any_success = False
            env_succs: List[float] = []
            env_rews: List[float] = []

            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0, created_env=None)
            for success, total_reward, episode in rollouts:
                any_success = any_success or bool(success)
                successes_all.append(float(success))
                env_succs.append(float(success))
                rewards_all.append(float(total_reward))
                env_rews.append(float(total_reward))

            per_env_success_rates[env_name] = float(np.mean(env_succs)) if len(env_succs) > 0 else 0.0
            per_env_any_success.append(any_success)

        output: Dict[str, Any] = {}
        output['rollout'] = {
            'overall_success_rate': float(np.mean(successes_all)) if len(successes_all) > 0 else 0.0,
            'overall_average_reward': float(np.mean(rewards_all)) if len(rewards_all) > 0 else 0.0,
            'environments_solved': int(np.sum(per_env_any_success)),
            'rollout_count': len(successes_all),
        }
        output['rollout_success_rate'] = {k: float(v) for k, v in per_env_success_rates.items()}
        return output

    def create_env(self, env_name: str):
        # 基于 LIBERO bddl 创建并行环境
        bench_dict = libero_benchmark.get_benchmark_dict()
        # 尝试不同的键名格式：原名、小写、去掉前缀
        benchmark_key = None
        for key in [self.benchmark_name, self.benchmark_name.lower(), 
                   self.benchmark_name.replace('LIBERO_', '').lower()]:
            if key in bench_dict:
                benchmark_key = key
                break
        
        if benchmark_key is None:
            raise KeyError(f"Benchmark '{self.benchmark_name}' not found. Available: {list(bench_dict.keys())}")
        
        bench = bench_dict[benchmark_key]()
        self._bench = bench
        self.env_names = bench.get_task_names()
        task_id = self.env_names.index(env_name)
        task = bench.get_task(task_id)
        bddl = get_libero_path("bddl_files")
        bddl_file = f"{bddl}/{task.problem_folder}/{task.bddl_file}"

        def _env_factory():
            return OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)

        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        env = SubprocVectorEnv([_env_factory for _ in range(env_num)])
        return env, task_id, env_num

    def run_policy_in_env(
        self,
        env_name: str,
        policy,
        all_init_states: Any = None,
        render: bool = False,
        created_env: Any = None,
        random_init: bool = False,
    ) -> Iterable[Tuple[bool, float, Dict[str, Any]]]:
        # 真环境交互（简化版）
        env, env_id, env_num = created_env if created_env is not None else self.create_env(env_name)
        bench = getattr(self, "_bench", None)
        if bench is None:
            bench_dict = libero_benchmark.get_benchmark_dict()
            bench = bench_dict[self.benchmark_name]()
        task_id = self.env_names.index(env_name)
        task = bench.get_task(task_id)
        task_description = task.language

        total_loops = (self.rollouts_per_env + env_num - 1) // env_num

        # 将 policy 的 norm stats 注入 runner
        try:
            self.state_mean = getattr(policy, 'state_mean').detach().cpu().numpy()
            self.state_std = getattr(policy, 'state_std').detach().cpu().numpy()
            self.action_mean = getattr(policy, 'action_mean').detach().cpu().numpy()
            self.action_std = getattr(policy, 'action_std').detach().cpu().numpy()
        except Exception:
            pass

        for loop_idx in range(total_loops):
            # 取 init states
            if all_init_states is None:
                all_init_states = bench.get_task_init_states(task_id)
            indices = np.arange(loop_idx * env_num, (loop_idx + 1) * env_num) % all_init_states.shape[0]
            init_states_ = all_init_states[indices]

            env.reset()
            try:
                env.set_init_state(init_states_)
            except Exception:
                pass

            # 回放缓存
            action_queues = [deque(maxlen=50) for _ in range(env_num)]
            done_flags = [False] * env_num
            success_flags = [False] * env_num
            total_reward = [0.0] * env_num

            episode = {
                'actions': [],
                'valid': [],
                'actions_normalized': [],
                'log_prob': [],
                'observations': [],
                'task_description': [],
            }

            t = 0
            while t < (self.max_episode_length + self.num_steps_wait):
                # wait steps
                if t < self.num_steps_wait:
                    dummy = [0.0]*6 + [-1.0]
                    obs, rew, done, info = env.step([dummy for _ in range(env_num)])
                    t += 1
                    continue

                step_actions = [[0.0]*7 for _ in range(env_num)]
                step_valid = [False]*env_num
                step_norm_actions = [None]*env_num
                step_logp = [None]*env_num
                step_obs = [None]*env_num

                need_infer_ids = []
                obs_list = []
                for i in range(env_num):
                    if not done_flags[i]:
                        step_valid[i] = True
                        need_infer_ids.append(i)
                        # 构造 PI0 观测（使用当前 env obs[i]）
                        raw_i = obs[i]
                        obs_pi0 = build_pi0_observation(raw_i, task_description, self.state_mean, self.state_std)
                        obs_list.append(obs_pi0)
                        step_obs[i] = obs_pi0

                # 若需要新 chunk
                conduct_infer = all(len(action_queues[i]) == 0 for i in need_infer_ids)
                if conduct_infer and len(need_infer_ids) > 0:
                    # 拼批次
                    batch_obs = {
                        'image': {
                            'base_0_rgb': np.stack([o['image']['base_0_rgb'] for o in obs_list], axis=0),
                            'left_wrist_0_rgb': np.stack([o['image']['left_wrist_0_rgb'] for o in obs_list], axis=0),
                        },
                        'state': np.stack([o['state'] for o in obs_list], axis=0),
                        'prompt': [task_description]*len(obs_list),
                    }
                    _verbose = os.environ.get('PI0_VERBOSE', '0') == '1'
                    if _verbose:
                        print(f"[Runner] t={t} | need_infer={len(need_infer_ids)} | building batch...")
                    norm_actions = policy.select_action(batch_obs)
                    if _verbose:
                        try:
                            shape_info = tuple(norm_actions.shape) if hasattr(norm_actions, 'shape') else np.asarray(norm_actions).shape
                        except Exception:
                            shape_info = 'unknown'
                        print(f"[Runner] t={t} | chunk received: shape={shape_info}")
                    if hasattr(norm_actions, 'detach'):
                        norm_actions = norm_actions.detach().cpu().numpy()
                    # 逐 env 放入队列
                    for j, env_id in enumerate(need_infer_ids):
                        chunk = norm_actions[j]
                        # 反归一 + 绝对化
                        unnorm_state_t0 = extract_state_8d(obs[env_id])
                        abs_chunk = denorm_action_sequence(chunk, self.action_mean, self.action_std, unnorm_state_t0)
                        # 存入队列
                        for k in range(abs_chunk.shape[0]):
                            action_queues[env_id].append(abs_chunk[k])
                        # 记录一次 chunk（用于优化器样本构造）
                        episode['actions'].append(abs_chunk.tolist())
                        episode['valid'].append([True]*abs_chunk.shape[0])
                        episode['actions_normalized'].append(chunk.tolist())
                        episode['log_prob'].append(np.zeros_like(chunk).tolist())
                        episode['observations'].append(obs_list)  # 记录该次推理使用的观测
                        episode['task_description'].append([task_description]*len(step_actions))

                # 执行一步（从队列里取一帧动作）
                for i in range(len(need_infer_ids)):
                    env_id = need_infer_ids[i]
                    if len(action_queues[env_id]) > 0:
                        step_actions[env_id] = action_queues[env_id].popleft().tolist()

                obs, rew, done, info = env.step(step_actions)
                for i in range(env_num):
                    total_reward[i] += float(rew[i]) if isinstance(rew, (list, tuple)) else float(rew)
                    done_flags[i] = bool(done[i]) if isinstance(done, (list, tuple)) else bool(done)
                    success_flags[i] = success_flags[i] or done_flags[i]

                if all(done_flags):
                    break
                t += 1

            # 逐 env 产出一次 rollouts（平均 reward / success）
            for k in range(env_num):
                yield success_flags[k], total_reward[k], episode

        if created_env is None:
            try:
                env.close()
            except Exception:
                pass


