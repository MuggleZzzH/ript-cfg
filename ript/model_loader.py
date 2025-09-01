import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import instantiate
import ript.utils.utils as utils
from ript.algos.rl_optimizers import QuestModelAdapter

def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint and handle module prefixes
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        Dictionary with model state dict
    """
    checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    print(f'loading from checkpoint {checkpoint_path}')
    state_dict = utils.load_state(checkpoint_path)
    loaded_state_dict = {}
    for key, value in state_dict['model'].items():
        if 'module.' in key:
            loaded_state_dict[key[7:]] = value
        else:
            loaded_state_dict[key] = value
    return loaded_state_dict

def load_quest_model(cfg, local_tasks, device, device_id=None, world_size=1, use_ddp=False):
    """
    Load Quest model, reference model, optimizer and scheduler
    
    Args:
        cfg: Configuration object
        local_tasks: List of tasks for this process
        device: CUDA device
        device_id: Local CUDA device ID (only needed with DDP)
        world_size: Number of distributed processes
        use_ddp: Whether to wrap model in DDP
        
    Returns:
        Tuple of (model, refer_model, model_adapter, optimizers, schedulers)
    """
    # Create main model
    print(f'Creating policy model on {device}')
    
    # Initialize model
    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.device = device
    model.train()
    
    # Wrap with DDP if needed
    if use_ddp:
        if device_id is None:
            device_id = int(device.split(':')[1]) if ':' in device else 0
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
        model_unwrapped = model.module
    else:
        model_unwrapped = model

    checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        print(f'Loading policy model from checkpoint: {checkpoint_path}')
        loaded_state_dict = load_checkpoint(checkpoint_path)
        utils.soft_load_state_dict(model_unwrapped, loaded_state_dict)
        print('Loaded policy model from checkpoint')
    else:
        print('Starting from scratch')
    
    # Create optimizer and scheduler
    optimizers = model_unwrapped.get_optimizers()
    schedulers = model_unwrapped.get_schedulers(optimizers)
    
    # Create model adapter
    model_adapter = QuestModelAdapter(model_unwrapped)
    
    return model, model_adapter, optimizers, schedulers 