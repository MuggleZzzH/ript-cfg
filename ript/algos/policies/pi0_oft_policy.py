"""
PI0 policy wrapper integrated with openpi_pytorch.pi0.PI0Policy.

Contract:
- self.model: underlying PI0Policy (optionally DDP-wrapped by train script)
- self.trainable_params:
    - 'model': main body params (excludes a small "header" subset)
    - 'header': small, meaningful subset (action_out_proj) for optional second optim group
- select_action(): returns normalized actions (B, 50, 7). Runner做反归一化与绝对化。

Note on "header":
- 之前用过 dummy 线性层占位，为兼容双优化器脚本。现改为使用 PI0 真正的动作输出头 `action_out_proj`，更合理。
- 若该模块不可用，自动回退到单组参数（不创建 header 组）。
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


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
from pi0.modeling_pi0 import PI0Policy  # type: ignore


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
                # 直接使用PI0模型目录中的tokenizer（更加匹配）
                self.model = PI0Policy.from_pretrained(
                    pretrained_path,
                    tokenizer_path=str(pretrained_path),
                    local_files_only=True  # 确保完全本地加载，不联网
                )
            else:
                # 冷启动（无预训练权重时）
                self.model = PI0Policy(PI0Policy.config_class())
        except Exception as e:
            raise RuntimeError(f"Failed to load PI0Policy: {e}")

        # 允许通过 kwargs 配置条件注入模式（bias/concat/token）
        condition_mode = kwargs.get("condition_mode", None)
        if condition_mode is not None:
            try:
                if hasattr(self.model, "model") and hasattr(self.model.model, "condition_mode"):
                    self.model.model.condition_mode = condition_mode
                elif hasattr(self.model, "condition_mode"):
                    setattr(self.model, "condition_mode", condition_mode)
            except Exception:
                pass

        self.model.to(self.device)
        self.model.eval()

        # 2.1) Optional: apply freeze / expert-only flags from kwargs (works with pretrained too)
        try:
            freeze_flag = kwargs.get("freeze_vision_encoder", None)
            expert_only = kwargs.get("train_expert_only", None)
            core = self.model.model if hasattr(self.model, "model") else self.model
            # Update top-level config if present
            if hasattr(core, "config"):
                if freeze_flag is not None:
                    setattr(core.config, "freeze_vision_encoder", bool(freeze_flag))
                if expert_only is not None:
                    setattr(core.config, "train_expert_only", bool(expert_only))
            # Update expert module and re-apply requires_grad
            if hasattr(core, "paligemma_with_expert"):
                pe = core.paligemma_with_expert
                if hasattr(pe, "config"):
                    if freeze_flag is not None:
                        setattr(pe.config, "freeze_vision_encoder", bool(freeze_flag))
                    if expert_only is not None:
                        setattr(pe.config, "train_expert_only", bool(expert_only))
                if hasattr(pe, "set_requires_grad"):
                    pe.set_requires_grad()
        except Exception:
            pass

        # 2) Trainable parameter groups (meaningful header):
        #    - 将 PI0 的动作输出投影头(action_out_proj)作为 header 组
        #    - 其余参数归为 model 组；若未找到该模块，退化为仅 model 组
        header_params: list[nn.Parameter] = []
        try:
            # unwrap if needed
            core = self.model.model if hasattr(self.model, "model") else self.model
            if hasattr(core, "action_out_proj"):
                header_params = [p for p in core.action_out_proj.parameters() if p.requires_grad]
        except Exception:
            header_params = []

        # exclude header params from main group (by id)
        header_ids = {id(p) for p in header_params}
        model_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in header_ids]

        self.trainable_params = {"model": model_params}
        if len(header_params) > 0:
            self.trainable_params["header"] = header_params

        self.ddp_wrapped = False
        if ddp_wrap and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device_id])
            self.ddp_wrapped = True
            self.model.eval()

        # 3) Simple cfg holder for compatibility
        class _Cfg:
            def __init__(self) -> None:
                self.use_film = False
                self.log_scale_clip = None

        self.cfg = _Cfg()

        # 4) Load normalization stats (state[:8], action[:7])
        self._load_norm_stats(self.norm_stats_path)

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

    @property
    def core_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    # ===== Inference =====
    @torch.inference_mode()
    def select_action(self, observation: Dict[str, Any], cfg_scale: float = 1.0, is_positive_infer: Optional[int] = None):
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

        # HWC uint8 -> BCHW uint8 tensor  
        def to_nchw_uint8(x):
            if isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.as_tensor(x)
            if t.ndim == 3:  # (H, W, C) -> add batch dim -> (1, C, H, W)
                t = t.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
            elif t.ndim == 4:  # Already (B, H, W, C) -> (B, C, H, W)
                t = t.permute(0, 3, 1, 2)
            return t.to(device=device, dtype=torch.uint8)

        batch["image"]["base_0_rgb"] = to_nchw_uint8(base)
        if wrist is not None:
            batch["image"]["left_wrist_0_rgb"] = to_nchw_uint8(wrist)

        state = observation["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        state = state.to(device=device, dtype=torch.float32)
        
        # 确保 state 有 batch 维度：(8,) -> (1, 8)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch["state"] = state

        # 语言
        if batch["prompt"] is None:
            batch["prompt"] = [""] * batch["state"].shape[0]

        # 可控轻量日志
        _verbose = os.environ.get("PI0_VERBOSE", "0") == "1"
        if _verbose:
            bsz = batch["state"].shape[0]
            h, w = batch["image"]["base_0_rgb"].shape[-2:]
            print(f"[PI0] infer start | B={bsz}, HxW={h}x{w}")
        start_ts = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_ts = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_ts is not None:
            start_ts.record()

        # PI0 前向：得到 (B, 50, D)，截取前7维为动作维度
        actions = self.core_model.select_action(batch, cfg_scale=cfg_scale, is_positive_infer=is_positive_infer)

        if end_ts is not None:
            end_ts.record()
            torch.cuda.synchronize()
            ms = start_ts.elapsed_time(end_ts)
            if _verbose:
                print(f"[PI0] infer done  | latency={ms:.1f} ms")
        if isinstance(actions, torch.Tensor):
            act = actions
        else:
            act = torch.as_tensor(actions)
        act = act.to(device=device, dtype=torch.float32)

        # 仅取前 7 维作为可执行动作（标准化）
        act = act[..., :7]
        if _verbose:
            print(f"[PI0] action shape={tuple(act.shape)} (B,50,7)")
        return act

