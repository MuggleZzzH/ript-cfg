"""
PI0 policy wrapper integrated with openpi_pytorch.pi0.PI0Policy.

Contract:
- self.model: underlying PI0Policy (optionally DDP-wrapped by train script)
- self.trainable_params: {'model': [...], 'header': [...]} for two optim groups
- select_action(): returns normalized actions (B, 50, 7). Runner做反归一化与绝对化。
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path
import os
import sys

import torch
from torch import nn


def _ensure_openpi_on_path(workspace_root: Optional[str] = None) -> None:
    """Best-effort to make openpi_pytorch importable when running inside this repo layout."""
    try:
        import openpi_pytorch  # noqa: F401
        return
    except Exception:
        pass
    # Try to add workspace root to sys.path so that 'openpi_pytorch' becomes importable
    if workspace_root is None:
        # Assume this file is under .../ript-vla_ori/ript/algos/policies/
        workspace_root = str(Path(__file__).resolve().parents[3])
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)


_ensure_openpi_on_path()
from openpi_pytorch.pi0.modeling_pi0 import PI0Policy  # type: ignore


class PI0_OFT_Policy:
    def __init__(
        self,
        device_id: int = 0,
        norm_stats_path: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        ddp_wrap: bool = True,
        **kwargs: Any,
    ) -> None:
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.norm_stats_path = norm_stats_path

        # 1) Load PI0 policy (默认路径可后续在服务器替换)
        try:
            if pretrained_path and Path(pretrained_path).exists():
                self.model = PI0Policy.from_pretrained(pretrained_path)
            else:
                # 冷启动（无预训练权重时）
                self.model = PI0Policy(PI0Policy.config_class())
        except Exception as e:
            raise RuntimeError(f"Failed to load PI0Policy: {e}")

        self.model.to(self.device)
        self.model.eval()

        # 2) Dummy heads to satisfy train script's optimizer/header saving contract
        #    - 保证 header 组非空，避免 AdamW 空参数错误
        self.action_head = nn.Linear(1, 1, bias=False).to(self.device)
        self.scale_head = nn.Linear(1, 1, bias=False).to(self.device)

        # 3) Trainable parameter groups
        self.trainable_params = {
            "model": [p for p in self.model.parameters() if p.requires_grad],
            "header": list(self.action_head.parameters()) + list(self.scale_head.parameters()),
        }

        # 4) Simple cfg holder for compatibility
        class _Cfg:
            def __init__(self) -> None:
                self.use_film = False
                self.log_scale_clip = None

        self.cfg = _Cfg()

        # 5) Load normalization stats (state[:8], action[:7])
        self._load_norm_stats(self.norm_stats_path)

        # 6) Optionally DDP-wrapped by external script. Here we keep plain module.

    # ===== Normalization =====
    def _load_norm_stats(self, norm_stats_path: Optional[str]) -> None:
        self._norm_loaded = False
        self.state_mean = torch.zeros(8, dtype=torch.float32, device=self.device)
        self.state_std = torch.ones(8, dtype=torch.float32, device=self.device)
        self.action_mean = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.action_std = torch.ones(7, dtype=torch.float32, device=self.device)

        if not norm_stats_path:
            return
        try:
            with open(norm_stats_path, "r") as f:
                data = json.load(f)
            sm = data["norm_stats"]["state"]["mean"][:8]
            ss = data["norm_stats"]["state"]["std"][:8]
            am = data["norm_stats"]["actions"]["mean"][:7]
            as_ = data["norm_stats"]["actions"]["std"][:7]
            self.state_mean = torch.tensor(sm, dtype=torch.float32, device=self.device)
            self.state_std = torch.tensor(ss, dtype=torch.float32, device=self.device)
            self.action_mean = torch.tensor(am, dtype=torch.float32, device=self.device)
            self.action_std = torch.tensor(as_, dtype=torch.float32, device=self.device)
            self._norm_loaded = True
        except Exception as e:
            # 使用默认单位归一化
            print(f"[PI0_OFT_Policy] Warning: Failed to load norm_stats from {norm_stats_path}: {e}")

    # ===== Inference =====
    @torch.inference_mode()
    def select_action(self, observation: Dict[str, Any], cfg_scale: float = 1.0):
        """
        返回标准化动作序列 (B, 50, 7)。
        - 输入 observation:
          {
            'image': {'base_0_rgb': uint8(B,H,W,3), 'left_wrist_0_rgb': uint8(B,H,W,3)},
            'state': float32 (B, 8)  已按 norm_stats 归一化,
            'prompt': [str, ...]
          }
        - runner 负责反归一化与绝对化（加回 eef pose 偏置）。
        """
        # 设备与 dtype
        device = self.device

        # 将 numpy/CPU 数据搬到 device，保持 PI0 期望类型
        batch = {
            "image": {},
            "state": None,
            "prompt": observation.get("prompt", None),
        }

        base = observation["image"]["base_0_rgb"]
        wrist = observation["image"].get("left_wrist_0_rgb", None)

        # HWC uint8 -> NCHW uint8 tensor
        def to_nchw_uint8(x):
            if isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.as_tensor(x)
            if t.ndim == 3:  # (H, W, C)
                t = t.permute(2, 0, 1)
            return t.to(device=device, dtype=torch.uint8)

        batch["image"]["base_0_rgb"] = to_nchw_uint8(base)
        if wrist is not None:
            batch["image"]["left_wrist_0_rgb"] = to_nchw_uint8(wrist)

        state = observation["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        batch["state"] = state.to(device=device, dtype=torch.float32)

        # 语言
        if batch["prompt"] is None:
            batch["prompt"] = [""] * batch["state"].shape[0]

        # PI0 前向：得到 (B, 50, D)，截取前7维为动作维度
        actions = self.model.select_action(batch)
        if isinstance(actions, torch.Tensor):
            act = actions
        else:
            act = torch.as_tensor(actions)
        act = act.to(device=device, dtype=torch.float32)

        # 仅取前 7 维作为可执行动作（标准化）
        act = act[..., :7]
        return act



