"""
CFG-Flow optimizer for PI0 (chunk-based FM loss with RLOO advantage weighting).

Implements:
- Rollout → per-episode rewards → RLOO advantages
- Episode → window samples (chunk=50) with normalized relative actions
- Shared-noise/time dual-branch (pos/uncond) loss skeleton (is_positive field passed through)
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import numpy as np


class CFGFlowOptimizerPI0:
    def __init__(
        self,
        rollout_generator,
        reward_function,
        gradient_accumulation_steps: int = 1,
        alpha_uncond: float = 0.0,
        cf_dropout_p: float = 0.1,
        stride: int = 1,
        max_windows_per_episode: int | None = None,
        optimizer_batch_size: int = 4,
        grad_norm_clip_model: float = 1.0,
        grad_norm_clip_header: float = 1.0,
        use_binary_advantage: bool = True,  # 开关：True=二值化(类似IQL)，False=连续advantage
        **kwargs: Any,
    ) -> None:
        self.rollout_generator = rollout_generator
        self.reward_function = reward_function
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.alpha_uncond = alpha_uncond
        self.cf_dropout_p = max(0.0, min(1.0, cf_dropout_p))
        self.stride = max(1, stride)
        self.max_windows_per_episode = max_windows_per_episode
        self.optimizer_batch_size = max(1, optimizer_batch_size)
        self.grad_norm_clip_model = grad_norm_clip_model
        self.grad_norm_clip_header = grad_norm_clip_header
        self.use_binary_advantage = use_binary_advantage

    # ====== Public API ======
    def optimize(self, model, batch, optimizers, data_iterator=None, dataloader=None) -> Dict[str, float]:
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 1) Generate rollouts
        episodes, task_ids, valid_mask, samples_checked = self.rollout_generator.generate_rollouts(
            model, batch, data_iterator, dataloader
        )

        # 2) Compute rewards per episode
        scores: List[float] = []
        for i in range(len(episodes)):
            r = self.reward_function.compute_reward(i, episodes[i], batch)
            scores.append(float(r))
        scores_t = torch.tensor(scores, device=device, dtype=torch.float32)

        # 3) RLOO advantages (group by demo)
        demo_batch_size = max(1, len(episodes) // max(1, self.rollout_generator.rloo_batch_size))
        rloo_batch_size = max(1, self.rollout_generator.rloo_batch_size)
        if demo_batch_size * rloo_batch_size != len(episodes):
            # pad to multiple if needed (rare)
            pad = demo_batch_size * rloo_batch_size - len(episodes)
            if pad > 0:
                scores_t = torch.nn.functional.pad(scores_t, (0, pad), value=scores_t[-1].item())
        rewards = scores_t.view(demo_batch_size, rloo_batch_size)
        # leave-one-out baseline per demo row
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / max(1, (rloo_batch_size - 1))
        adv = (rewards - baseline).reshape(-1)
        adv = adv[: len(episodes)]

        # 4) Build window samples (obs[t] → actions[t:t+50])
        samples, sample_adv = self._episodes_to_window_samples(episodes, adv, device)
        if len(samples) == 0:
            return {
                "mean_scores": float(scores_t.mean().item()),
                "mean_advantage": float(adv.mean().item()),
                "non_zero_adv_ratio": float((adv != 0).float().mean().item()),
                "rollout_checked": float(samples_checked),
            }

        # 4.5) Global shuffle across window samples (keep per-sample advantage aligned)
        try:
            perm = torch.randperm(len(samples), device=device)
            samples = [samples[i] for i in perm.tolist()]
            if isinstance(sample_adv, torch.Tensor) and sample_adv.numel() == len(perm):
                sample_adv = sample_adv[perm]
        except Exception:
            # Fallback to CPU permutation if device randperm fails for any reason
            import numpy as _np
            _idx = _np.random.permutation(len(samples)).tolist()
            samples = [samples[i] for i in _idx]
            if isinstance(sample_adv, torch.Tensor) and sample_adv.numel() == len(_idx):
                sample_adv = sample_adv[_idx]

        # 5) Micro-batch training to avoid OOM
        total_samples = len(samples)
        accum_count = 0
        running_loss = 0.0
        for start in range(0, total_samples, self.optimizer_batch_size):
            end = min(start + self.optimizer_batch_size, total_samples)
            sub_samples = samples[start:end]
            sub_adv = sample_adv[start:end]

            batch_data = self._collate_samples(sub_samples, device)
            B, T, _ = batch_data["action"].shape
            D_max = int(getattr(getattr(model, "model", None), "config", None).max_action_dim) if hasattr(getattr(model, "model", None), "config") else 32
            noise = torch.randn(B, T, D_max, device=device, dtype=batch_data["state"].dtype)
            time = torch.rand(B, device=device, dtype=batch_data["state"].dtype) * 0.999 + 0.001

            batch_pos = dict(batch_data)
            batch_pos["noise"] = noise
            batch_pos["time"] = time
            batch_pos["is_positive"] = torch.ones(B, device=device, dtype=torch.long)
            loss_pos, loss_dict_pos = self._forward_losses(model, batch_pos)
            per_step_pos = loss_dict_pos["losses"].mean(dim=-1)

            batch_uncond = dict(batch_data)
            batch_uncond["noise"] = noise
            batch_uncond["time"] = time
            # 使用“∅无条件”分支：不注入条件（对齐CFGRL的unc分支语义）
            batch_uncond["is_positive"] = None
            _, loss_dict_uncond = self._forward_losses(model, batch_uncond)
            per_step_uncond = loss_dict_uncond["losses"].mean(dim=-1)

            mask = (~batch_data["action_is_pad"]).float()

            # 1) 二值优势标签（窗口级）
            is_good = (sub_adv >= 0).to(device=device, dtype=torch.float32)  # (B,)
            # 2) CF Dropout 采样
            do_cfg = torch.bernoulli(torch.full((B,), self.cf_dropout_p, device=device, dtype=torch.float32))  # (B,)
            # 3) 门控：cond 分支在"好样本且未dropout"时被选择
            m_cond = (is_good * (1.0 - do_cfg)).view(B, 1).expand_as(mask)    # (B, T)
            m_unc  = (1.0 - m_cond)                                           # (B, T)

            combined = m_cond * per_step_pos + m_unc * per_step_uncond
            valid_steps = mask.sum(dim=1).clamp_min(1.0)
            window_losses = (combined * mask).sum(dim=1) / valid_steps
            sub_loss = window_losses.mean()
            
            # 保存变量供监控使用
            if not hasattr(self, '_last_do_cfg'):
                self._last_do_cfg = do_cfg
                self._last_m_cond = m_cond
                self._last_m_unc = m_unc

            sub_loss_norm = sub_loss / max(1, self.gradient_accumulation_steps)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                sub_loss_norm.backward()
            running_loss += float(sub_loss.detach().item())
            accum_count += 1

            if accum_count % self.gradient_accumulation_steps == 0 or end == total_samples:
                if dist.is_initialized():
                    for group_name in ("model", "header"):
                        if hasattr(model, 'trainable_params') and group_name in model.trainable_params:
                            for p in model.trainable_params[group_name]:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                for opt in optimizers:
                    params = []
                    for group in opt.param_groups:
                        params.extend(group["params"])
                    
                    # 确定梯度裁剪值：区分model和header优化器
                    if hasattr(model, 'trainable_params'):
                        # 检查优化器参数是否属于header组
                        is_header_opt = False
                        if 'header' in model.trainable_params:
                            header_params = set(model.trainable_params['header'])
                            if any(param in header_params for param in params):
                                is_header_opt = True
                        
                        grad_clip_value = self.grad_norm_clip_header if is_header_opt else self.grad_norm_clip_model
                    else:
                        # 默认使用model梯度裁剪值
                        grad_clip_value = self.grad_norm_clip_model
                    
                    torch.nn.utils.clip_grad_norm_(params, grad_clip_value)
                    opt.step()
                    opt.zero_grad()
                accum_count = 0

        avg_loss = running_loss / max(1, (total_samples + self.optimizer_batch_size - 1) // self.optimizer_batch_size)

        metrics = {
            "training/fm_loss": float(avg_loss),
            "mean_scores": float(scores_t.mean().item()),
            "mean_advantage": float(adv.mean().item()),
            "non_zero_adv_ratio": float((adv != 0).float().mean().item()),
            "rollout_checked": float(samples_checked),
            "training/cf_dropout_rate": float(torch.mean(self._last_do_cfg).item()) if hasattr(self, '_last_do_cfg') else 0.0,
            "training/pos_rate": float(torch.mean((adv >= 0).float()).item()),
            "training/branch_usage/cond": float(torch.mean(self._last_m_cond.float()).item()) if hasattr(self, '_last_m_cond') else 0.0,
            "training/branch_usage/uncond": float(torch.mean(self._last_m_unc.float()).item()) if hasattr(self, '_last_m_unc') else 0.0,
        }
        # DDP metrics reduce
        if dist.is_initialized():
            for k, v in metrics.items():
                t = torch.tensor(v, device=device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                metrics[k] = float(t.item())

        return metrics

    # ====== Helpers ======
    def _episodes_to_window_samples(self, episodes: List[Dict[str, Any]], advantages: torch.Tensor, device: torch.device) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        samples: List[Dict[str, Any]] = []
        sample_adv: List[float] = []
        H = 50
        for ep_idx, ep in enumerate(episodes):
            # Prefer normalized relative actions if provided
            # episode['actions_normalized'] is expected as per openvla runner; fallback to compute from absolute if needed
            if 'actions_normalized' in ep and len(ep['actions_normalized']) > 0:
                # Our stub uses chunk-level list [ (T,7) ]
                if isinstance(ep['actions_normalized'][0], list):
                    act_norm = np.asarray(ep['actions_normalized'][0], dtype=np.float32)
                else:
                    act_norm = np.asarray(ep['actions_normalized'], dtype=np.float32)
            else:
                # Fallback: derive relative normalized actions from absolute if stats and states available (not covered here)
                continue

            # observations: take the starting observation for the window; our stub stores [[obs]]
            obs_list = ep.get('observations', [[]])
            if len(obs_list) == 0:
                continue
            # observations 通常为按时间步的列表；每个时间步是并行环境数量的列表
            # 兼容以下两种结构：
            # 1) [ {obs_dict_t0_env0}, {obs_dict_t1_env0}, ... ]
            # 2) [ [obs_dict_t0_env0, obs_dict_t0_env1, ...], [obs_dict_t1_env0, ...], ... ]
            first = obs_list[0]
            if isinstance(first, dict):
                obs0 = first
            elif isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
                obs0 = first[0]
            else:
                continue
            # valid mask
            valid = ep.get('valid', [[True] * len(act_norm)])
            valid_seq = np.asarray(valid[0], dtype=bool)

            T_seq = act_norm.shape[0]
            max_windows = self.max_windows_per_episode or max(1, (T_seq - H) // self.stride + 1)
            count = 0
            for start in range(0, max(0, T_seq - H + 1), self.stride):
                if count >= max_windows:
                    break
                end = start + H
                window_actions = act_norm[start:end]
                # pad if needed
                if window_actions.shape[0] < H:
                    pad = H - window_actions.shape[0]
                    window_actions = np.pad(window_actions, ((0, pad), (0, 0)), mode='constant')
                action_is_pad = np.ones((H,), dtype=bool)
                action_is_pad[: min(H, T_seq - start)] = False

                # build sample
                try:
                    img_base = obs0['image']['base_0_rgb']
                    img_wrist = obs0['image'].get('left_wrist_0_rgb', img_base)
                    state_np = np.asarray(obs0['state'], dtype=np.float32)
                    prompt_list = obs0.get('prompt', [''])
                    if isinstance(prompt_list, list):
                        prompt_out = prompt_list[0:1]
                    else:
                        prompt_out = [str(prompt_list)]
                except Exception:
                    continue

                sample = {
                    'image': {
                        'base_0_rgb': img_base,
                        'left_wrist_0_rgb': img_wrist,
                    },
                    'state': state_np,
                    'prompt': prompt_out,
                    'action': window_actions.astype(np.float32),
                    'action_is_pad': action_is_pad,
                }
                samples.append(sample)
                sample_adv.append(float(advantages[ep_idx].item()))
                count += 1

        if len(samples) == 0:
            return [], torch.empty(0, device=device)
        return samples, torch.tensor(sample_adv, device=device, dtype=torch.float32)

    def _collate_samples(self, samples: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        # images to NCHW uint8
        def to_nchw_uint8(x: np.ndarray) -> torch.Tensor:
            arr = np.asarray(x)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr.transpose(2, 0, 1)
            return torch.from_numpy(arr).to(device=device, dtype=torch.uint8)

        B = len(samples)
        base_imgs = torch.stack([to_nchw_uint8(s['image']['base_0_rgb']) for s in samples], dim=0)
        wrist_imgs = torch.stack([to_nchw_uint8(s['image']['left_wrist_0_rgb']) for s in samples], dim=0)
        states = torch.from_numpy(np.stack([s['state'] for s in samples], axis=0)).to(device=device, dtype=torch.float32)
        actions = torch.from_numpy(np.stack([s['action'] for s in samples], axis=0)).to(device=device, dtype=torch.float32)
        pads = torch.from_numpy(np.stack([s['action_is_pad'] for s in samples], axis=0)).to(device=device)
        prompts = [ (s.get('prompt', [''])[0] if isinstance(s.get('prompt', [''])[0], str) else str(s.get('prompt', [''])[0])) for s in samples ]

        batch = {
            'image': {
                'base_0_rgb': base_imgs,        # (B,3,H,W) uint8
                'left_wrist_0_rgb': wrist_imgs, # (B,3,H,W) uint8
            },
            'state': states,                    # (B,8)
            'action': actions,                  # (B,50,7) normalized relative
            'action_is_pad': pads,              # (B,50) bool
            'prompt': prompts,                  # list[str]
        }
        return batch

    def _forward_losses(self, model, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Call underlying PI0Policy forward to get per-step-per-dim losses in loss_dict['losses']
        # 用混合精度降低显存
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model.model.forward(batch)
        else:
            out = model.model.forward(batch)
        if isinstance(out, tuple):
            loss, loss_dict = out
        else:
            # Should not happen in PI0Policy
            loss, loss_dict = out, {}
        return loss, loss_dict


