"""
PI0 Libero runner with open-loop=50 execution and episode fields aligned with rollout_generator expectations.
"""

from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

from .pi0_runner_utils import build_pi0_observation, denorm_action_sequence, extract_state_8d


class Pi0LiberoRunner:
    def __init__(
        self,
        benchmark_name: str = "libero_spatial",
        rollouts_per_env: int = 1,
        num_parallel_envs: int = 1,
        max_episode_length: int | None = None,
        task_names_to_use: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.max_episode_length = max_episode_length or 300
        self.task_names_to_use = task_names_to_use or []

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
        # 与原Runner接口一致，这里仅返回占位（真实工程中应构造SubprocVectorEnv）
        class _StubEnv:
            def close(self):
                return None

        env = _StubEnv()
        env_id = 0
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        return env, env_id, env_num

    def run_policy_in_env(
        self,
        env_name: str,
        policy,
        all_init_states: Any = None,
        render: bool = False,
        created_env: Any = None,
        random_init: bool = False,
    ) -> Iterable[Tuple[bool, float, Dict[str, Any]]]:
        # 占位环境循环：实际应替换为 LIBERO env 交互。
        for _ in range(self.rollouts_per_env):
            success = False
            total_reward = 0.0

            # 伪观测与 init state
            raw_obs = {
                "agentview_image": np.ones((224, 224, 3), dtype=np.uint8) * 128,
                "robot0_eye_in_hand_image": np.ones((224, 224, 3), dtype=np.uint8) * 128,
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            unnorm_state_t0 = extract_state_8d(raw_obs)
            obs_pi0 = build_pi0_observation(raw_obs, env_name, self.state_mean, self.state_std)

            # 生成 50 步标准化动作
            norm_actions = policy.select_action(obs_pi0)  # (1,50,7) or (50,7)
            if hasattr(norm_actions, 'detach'):
                norm_actions = norm_actions.detach().cpu().numpy()
            norm_actions = norm_actions[0] if norm_actions.ndim == 3 else norm_actions

            # 反归一并转为绝对
            acts = denorm_action_sequence(norm_actions, self.action_mean, self.action_std, unnorm_state_t0)

            episode = {
                "actions": [acts.tolist()],
                "valid": [[True] * len(acts)],
                "actions_normalized": [norm_actions.tolist()],
                "log_prob": [np.zeros_like(norm_actions).tolist()],
                "observations": [[obs_pi0]],
                "task_description": [[env_name]],
            }
            yield success, total_reward, episode


