import os
import numpy as np

class Logger:
    """
    The purpose of this simple logger is to log intermittently and log average values since the last log
    """
    def __init__(self, log_interval, backend: str | None = None):
        self.log_interval = log_interval
        self.data = None
        # 支持按环境变量切换日志后端：wandb | swanlab | none
        self.backend = (backend or os.environ.get("LOG_BACKEND", "wandb")).lower()
        self._backend_module = None

    def _ensure_backend(self):
        if self._backend_module is not None:
            return
        try:
            if self.backend == "swanlab":
                import swanlab as _mod  # type: ignore
                self._backend_module = _mod
            elif self.backend == "wandb":
                import wandb as _mod  # type: ignore
                self._backend_module = _mod
            else:
                self._backend_module = None
        except Exception:
            self._backend_module = None

    def update(self, info, step):
        info = flatten_dict(info)
        if self.data is None:
            self.data = {key: [] for key in info}
        
        for key in info:
            self.data[key].append(info[key])
        
        if step % self.log_interval == 0:
            means = {key: np.mean(value) for key, value in self.data.items()}
            self.log(means, step)
            self.data = None

    def log(self, info, step):
        info_flat = flatten_dict(info)
        if self.backend == "wandb":
            self._ensure_backend()
            if self._backend_module is not None:
                try:
                    self._backend_module.log(info_flat, step=step)
                except Exception:
                    pass
        elif self.backend == "swanlab":
            self._ensure_backend()
            if self._backend_module is not None:
                try:
                    self._backend_module.log(info_flat, step=step)
                except Exception:
                    pass
        else:
            # backend=none 或不可用：静默丢弃
            pass


def flatten_dict(in_dict):
    """
    The purpose of this is to flatten dictionaries because as of writing wandb handling nested dicts is broken :( 
    https://community.wandb.ai/t/the-wandb-log-function-does-not-treat-nested-dict-as-it-describes-in-the-document/3330
    """

    out_dict = {}
    for key, value in in_dict.items():
        if type(value) is dict:
            for inner_key, inner_value in value.items():
                out_dict[f'{key}/{inner_key}'] = inner_value
        else:
            out_dict[key] = value
    return out_dict