"""
Utilities for PI0 runner: observation mapping and (de)normalization helpers.
"""

from typing import Any, Dict, Tuple
import numpy as np
import torch


def extract_state_8d(obs: Dict[str, Any]) -> np.ndarray:
    """Extract 8D state: 3 pos + 3 axis-angle + 2 gripper."""
    try:
        pos = np.asarray(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float32)[:3]
        quat = np.asarray(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=np.float32)[:4]
        try:
            # local fallback without external import
            axis = quat2axisangle_np(quat)
        except Exception:
            axis = np.zeros(3, dtype=np.float32)
        grip = np.asarray(obs.get("robot0_gripper_qpos", [0, 0]), dtype=np.float32)[:2]
        return np.concatenate([pos, axis, grip], dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def quat2axisangle_np(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q[3], q[0], q[1], q[2]
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w * w)
    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([x / s, y / s, z / s], dtype=np.float32)
    return (axis * angle).astype(np.float32)


def build_pi0_observation(raw_obs: Dict[str, Any], task_description: str, state_mean: np.ndarray, state_std: np.ndarray) -> Dict[str, Any]:
    base = raw_obs.get("agentview_image")
    wrist = raw_obs.get("robot0_eye_in_hand_image")
    if base is None:
        base = np.ones((224, 224, 3), dtype=np.uint8) * 128
    if wrist is None:
        wrist = np.ones((224, 224, 3), dtype=np.uint8) * 128

    # ensure HWC uint8
    def to_hwc_uint8(x):
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
            x = x.transpose(1, 2, 0)
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255).astype(np.uint8)
            else:
                x = x.astype(np.uint8)
        return x

    base = to_hwc_uint8(base)
    wrist = to_hwc_uint8(wrist)

    unnorm_state = extract_state_8d(raw_obs)
    state = (unnorm_state - state_mean) / (state_std + 1e-6)

    return {
        "image": {
            "base_0_rgb": base,
            "left_wrist_0_rgb": wrist,
        },
        "state": state.astype(np.float32),
        "prompt": [task_description],
    }


def denorm_action_sequence(norm_actions: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray, unnorm_state_t0: np.ndarray) -> np.ndarray:
    """Denormalize sequence (T,7) and convert delta joints (first 6 dims) to absolute by adding t0 pose.
    This mirrors the openpi_pytorch example logic.
    """
    acts = norm_actions * (action_std + 1e-6) + action_mean
    acts[:, :6] += unnorm_state_t0[:6]
    return acts



