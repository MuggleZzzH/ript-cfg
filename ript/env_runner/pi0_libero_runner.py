"""
PI0 Libero runner with open-loop=50 execution and episode fields aligned with rollout_generator expectations.
真实环境版（轻量）：创建 SubprocVectorEnv、使用 init_states、wait steps，然后以 50 步 chunk 执行。
"""

from typing import Any, Dict, Iterable, List, Tuple
import os
import numpy as np
import math
from collections import deque
from datetime import datetime
import imageio.v2 as imageio

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
        # 可用环境变量覆盖 n_video（便于脚本控制）
        try:
            n_video = int(os.environ.get('PI0_N_VIDEO', n_video))
        except Exception:
            pass
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

        # 准备视频输出目录
        video_dir = os.environ.get('PI0_VIDEO_DIR', os.path.join('output', 'videos'))
        if render:
            try:
                os.makedirs(video_dir, exist_ok=True)
            except Exception:
                render = False

        for loop_idx in range(total_loops):
            # 初始化处理
            disable_init_states = os.getenv("PI0_DISABLE_INIT_STATES", "0") == "1"
            if disable_init_states:
                # 快速测试：不设置初始状态
                try:
                    env.reset()
                except Exception:
                    pass
            else:
                # 正常：reset 后设置初始状态（与 OpenVLA 路径一致）
                # 取 init states
                if all_init_states is None:
                    all_init_states = bench.get_task_init_states(task_id)
                indices = np.arange(loop_idx * env_num, (loop_idx + 1) * env_num) % all_init_states.shape[0]
                init_states_ = all_init_states[indices]
                try:
                    env.reset()
                    if hasattr(env, 'set_init_state'):
                        env.set_init_state(init_states_)
                    elif hasattr(env, 'env_method'):
                        env.env_method('set_init_state', init_states_)
                except Exception as e:
                    print(f"[Pi0LiberoRunner] Warning: reset/set_init_state failed ({getattr(e, 'args', e)}); continuing with default reset only.")

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
            # 视频帧缓存（仅记录第0个并行环境，减少IO开销）
            frames: List[np.ndarray] = [] if render else None

            def _extract_frame_from_obs(o: Dict[str, Any]) -> np.ndarray | None:
                try:
                    # 兼容多种键名（不同版本 / 包装）
                    base = (
                        o.get('agentview_image',
                        o.get('agentview_rgb'))
                    )
                    wrist = (
                        o.get('robot0_eye_in_hand_image',
                        o.get('robot0_eye_in_hand',
                        o.get('eye_in_hand_rgb')))
                    )
                    if base is None or wrist is None:
                        return None
                    base = np.asarray(base)
                    wrist = np.asarray(wrist)
                    if base.ndim == 3 and base.dtype != np.uint8:
                        base = (base * 255.0).clip(0, 255).astype(np.uint8)
                    if wrist.ndim == 3 and wrist.dtype != np.uint8:
                        wrist = (wrist * 255.0).clip(0, 255).astype(np.uint8)
                    # 方向对齐（与预处理一致，180°翻转）
                    base = base[::-1, ::-1].copy()
                    wrist = wrist[::-1, ::-1].copy()
                    # 对齐尺寸
                    h = min(base.shape[0], wrist.shape[0])
                    w = min(base.shape[1], wrist.shape[1])
                    base = base[:h, :w]
                    wrist = wrist[:h, :w]
                    frame = np.concatenate([base, wrist], axis=1)
                    return frame
                except Exception:
                    return None

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
                    # 评测时可传 cfg_scale/is_positive_infer（如通过环境变量或 policy 属性控制）
                    cfg_scale = float(os.environ.get('PI0_CFG_SCALE', '1.0'))
                    is_pos_env = os.environ.get('PI0_IS_POSITIVE', None)
                    is_pos_flag = int(is_pos_env) if is_pos_env is not None else None
                    norm_actions = policy.select_action(batch_obs, cfg_scale=cfg_scale, is_positive_infer=is_pos_flag)
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
                    # Handle list/tuple/np.ndarray scalars robustly
                    if isinstance(rew, (list, tuple, np.ndarray)):
                        ri = rew[i]
                    else:
                        ri = rew
                    if isinstance(done, (list, tuple, np.ndarray)):
                        di = done[i]
                    else:
                        di = done
                    try:
                        total_reward[i] += float(ri)
                    except Exception:
                        # Fallback if ri is numpy scalar
                        total_reward[i] += float(np.asarray(ri).item())
                    done_flags[i] = bool(di)
                    success_flags[i] = success_flags[i] or done_flags[i]

                # 记录视频帧（仅并行环境0）
                if render and isinstance(obs, (list, tuple)) and len(obs) > 0 and frames is not None:
                    frame = _extract_frame_from_obs(obs[0])
                    if frame is not None:
                        frames.append(frame)

                if all(done_flags):
                    break
                t += 1

            # 保存视频（每个 loop 一段）
            if render and frames is not None and len(frames) > 0:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f'{env_name}_loop{loop_idx}_{ts}'
                out_mp4 = os.path.join(video_dir, base_name + '.mp4')
                _verbose = os.environ.get('PI0_VERBOSE', '0') == '1'
                if _verbose:
                    try:
                        print(f"[Runner] saving video: {out_mp4} | frames={len(frames)}")
                    except Exception:
                        pass
                try:
                    # 优先保存 mp4（需要 imageio-ffmpeg）
                    imageio.mimsave(out_mp4, frames, fps=10)
                except Exception:
                    # 若 mp4 失败（常见于缺少 ffmpeg），回退为 gif（无需外部二进制）
                    out_gif = os.path.join(video_dir, base_name + '.gif')
                    if _verbose:
                        try:
                            print(f"[Runner] mp4 failed; fallback to GIF: {out_gif}")
                        except Exception:
                            pass
                    try:
                        imageio.mimsave(out_gif, frames, duration=0.1)
                    except Exception:
                        if _verbose:
                            try:
                                print(f"[Runner] gif save failed; skipping video for this loop")
                            except Exception:
                                pass

            # 逐 env 产出一次 rollouts（平均 reward / success）
            for k in range(env_num):
                # 打包每个并行环境的 episode，并附带 success 字段供奖励函数使用
                episode_k = dict(episode)
                episode_k['success'] = bool(success_flags[k])
                yield success_flags[k], total_reward[k], episode_k

        if created_env is None:
            try:
                env.close()
            except Exception:
                pass