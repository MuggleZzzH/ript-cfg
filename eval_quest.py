import os
os.environ["NCCL_TIMEOUT"] = "108000"
import sys
import time
import hydra
import wandb
import datetime
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import default_collate
import ript.utils.utils as utils
from ript.utils.logger import Logger
from ript.utils.dist_utils import sync_rollout_results_via_file
from ript.algos.rl_optimizers import QuestModelAdapter, RolloutGenerator, RLOptimizer
from ript.model_loader import load_quest_model

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(config_path="config", version_base=None)
def main(cfg):
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # Initialize distributed training
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))
    
    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print('CUDA_VISIBLE_DEVICES:', cuda_visible_devices.split(','))
    device_number = cuda_visible_devices.split(',')[device_id] if cuda_visible_devices else str(device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_number
    print('device_id', device_id)
    print(f'rank {rank} CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Split tasks across GPUs
    all_tasks = cfg.task.task_names_to_use

    if all_tasks is None:
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(cfg.task.benchmark_name.lower())()
        all_tasks = benchmark.get_task_names()
        print('using all tasks from benchmark', benchmark.name)

    rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
    for task_i, task_name in enumerate(all_tasks):
        rank_to_tasks[task_i % world_size].append(task_name)
    local_tasks = rank_to_tasks[rank]

    # Print from all ranks with synchronization
    print(f'[RANK {rank}] World size: {world_size}, Device: {device}, Tasks: {local_tasks}\n')

    # Create model adapter
    model, _, _, _ = load_quest_model(
        cfg=cfg,
        local_tasks=local_tasks,
        device=device,
        device_id=device_id,
        world_size=world_size,
        use_ddp=True
    )

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    if rank == 0:
        print('Saving to:', experiment_dir)
        print('Experiment name:', experiment_name)

        wandb.init(
            dir=experiment_dir,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )

        logger = Logger(train_cfg.log_interval)
    else:
        logger = None
    

    # Set up environment runner
    num_parallel_envs = cfg.task.env_runner.num_parallel_envs
    env_runner = instantiate(cfg.task.env_runner, task_names_to_use=local_tasks, num_parallel_envs=num_parallel_envs, reset_type='ori')
    
    print(f'[RANK {rank}] Conducting rollout evaluation')
    rollout_results = env_runner.run(model.module, n_video=0, do_tqdm=train_cfg.use_tqdm) 
    print(f'[RANK {rank}] Rollout results: {rollout_results}')
    sync_rollout_results_via_file(rollout_results, logger, 0)
    
    # Synchronize all processes
    dist.barrier()
    
    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()