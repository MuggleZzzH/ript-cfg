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
        alpha_uncond: float = 0.1,
        stride: int = 1,
        max_windows_per_episode: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.rollout_generator = rollout_generator
        self.reward_function = reward_function
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.alpha_uncond = alpha_uncond
        self.stride = max(1, stride)
        self.max_windows_per_episode = max_windows_per_episode

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

        # 5) Collate batch tensors
        batch_data = self._collate_samples(samples, device)

        # 6) Shared-noise/time dual-branch FM loss (is_positive stub passed; model to consume later)
        B, T, D_label = batch_data["action"].shape
        D_max = int(getattr(getattr(model, "model", None), "config", None).max_action_dim) if hasattr(getattr(model, "model", None), "config") else 32
        noise = torch.randn(B, T, D_max, device=device, dtype=batch_data["state"].dtype)
        time = torch.rand(B, device=device, dtype=batch_data["state"].dtype) * 0.999 + 0.001

        # is_positive flags from advantages
        w_pos = (sample_adv > 0).float().to(device)

        # pos branch
        batch_pos = dict(batch_data)
        batch_pos["noise"] = noise
        batch_pos["time"] = time
        batch_pos["is_positive"] = torch.ones(B, device=device, dtype=torch.long)
        loss_pos, loss_dict_pos = self._forward_losses(model, batch_pos)
        per_step_pos = loss_dict_pos["losses"].mean(dim=-1)  # (B,T)

        # uncond branch (shared noise/time)
        batch_uncond = dict(batch_data)
        batch_uncond["noise"] = noise
        batch_uncond["time"] = time
        batch_uncond["is_positive"] = torch.zeros(B, device=device, dtype=torch.long)
        _, loss_dict_uncond = self._forward_losses(model, batch_uncond)
        per_step_uncond = loss_dict_uncond["losses"].mean(dim=-1)  # (B,T)

        # padding-aware reduction
        mask = (~batch_data["action_is_pad"]).float()  # (B,T)
        w_pos_exp = w_pos.view(B, 1).expand_as(mask)
        combined = w_pos_exp * per_step_pos + self.alpha_uncond * per_step_uncond
        valid_steps = mask.sum(dim=1).clamp_min(1.0)
        window_losses = (combined * mask).sum(dim=1) / valid_steps  # (B,)
        final_loss = window_losses.mean()

        # 7) Backward + step with gradient accumulation
        final_loss_norm = final_loss / max(1, self.gradient_accumulation_steps)
        final_loss_norm.backward()

        # sync grads (avg) in DDP if initialized
        if dist.is_initialized():
            for group_name in ("model", "header"):
                if hasattr(model, 'trainable_params') and group_name in model.trainable_params:
                    for p in model.trainable_params[group_name]:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # step all optimizers
        for opt in optimizers:
            # clip grads on parameters belonging to this optimizer
            params = []
            for group in opt.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

        metrics = {
            "training/fm_loss": float(final_loss.detach().item()),
            "mean_scores": float(scores_t.mean().item()),
            "mean_advantage": float(adv.mean().item()),
            "non_zero_adv_ratio": float((adv != 0).float().mean().item()),
            "rollout_checked": float(samples_checked),
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
            if len(obs_list) == 0 or len(obs_list[0]) == 0:
                continue
            obs0 = obs_list[0]
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
                sample = {
                    'image': {
                        'base_0_rgb': obs0['image']['base_0_rgb'],
                        'left_wrist_0_rgb': obs0['image'].get('left_wrist_0_rgb', obs0['image']['base_0_rgb']),
                    },
                    'state': np.asarray(obs0['state'], dtype=np.float32),
                    'prompt': obs0.get('prompt', [''])[0:1],
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
        out = model.model.forward(batch)
        if isinstance(out, tuple):
            loss, loss_dict = out
        else:
            # Should not happen in PI0Policy
            loss, loss_dict = out, {}
        return loss, loss_dict



