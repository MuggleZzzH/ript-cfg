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

def collate_fn_state(batch):
    # Process the special case first
    states = [item['init_state']['states'] for item in batch]
    max_len = max(s.shape[-1] for s in states)
    
    padded_states = []
    masks = []
    modified_batch = []
    
    for item in batch:
        # Pad states and create mask
        tensor = torch.as_tensor(item['init_state']['states']).float()
        pad_size = max_len - tensor.shape[-1]
        padded = torch.nn.functional.pad(tensor, (0, pad_size))
        padded_states.append(padded)
        
        mask = torch.ones(tensor.shape[-1], dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
        masks.append(mask)
        
        # Create a modified item without the special field
        modified_item = {key: item[key] for key in item.keys() if key != 'init_state'}
        modified_batch.append(modified_item)

    # Collate all other fields normally
    collated_batch = default_collate(modified_batch)
    
    # Add our processed states and mask back in
    collated_batch['init_state'] = {}
    collated_batch['init_state']['states'] = torch.stack(padded_states)
    collated_batch['init_state']['pad_mask'] = torch.stack(masks)
    
    return collated_batch

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


    # prepare iteration based training count
    total_examples = cfg.task.rollouts_per_env * len(all_tasks)
    steps_per_epoch = total_examples // cfg.train_dataloader.batch_size
    save_interval_steps = train_cfg.save_interval

    if train_cfg.n_steps != -1:
        total_steps = train_cfg.n_steps
    else:
        total_steps = steps_per_epoch * train_cfg.n_epochs
    
    if train_cfg.rollout_steps != -1:
        rollout_interval_steps = train_cfg.rollout_steps
    else:
        rollout_interval_steps = train_cfg.rollout_interval
    
    if rank == 0:
        print('Training Configuration:')
        print(f"\tTotal Examples per Epoch: {total_examples}")
        print(f"\tSteps per Epoch: {steps_per_epoch}")
        print(f"\tTotal Steps: {total_steps}")
        print(f"\tSave Interval Steps: {save_interval_steps}")
        print(f"\tRollout Interval Steps: {rollout_interval_steps}")


    # setup dataset and dataloader
    dataset = instantiate(cfg.task.dataset, task_names_to_use=local_tasks)

    # Load model using the extracted function
    # Create model adapter
    model, model_adapter, optimizers, schedulers = load_quest_model(
        cfg=cfg,
        local_tasks=local_tasks,
        device=device,
        device_id=device_id,
        world_size=world_size,
        use_ddp=True
    )

    model.module.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset,
        collate_fn=collate_fn_state,
        batch_size=cfg.train_dataloader.batch_size // world_size
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
    if len(local_tasks) >= 4:
        num_parallel_envs = 1
    else:
        num_parallel_envs = cfg.task.env_runner.num_parallel_envs
    env_runner = instantiate(cfg.task.env_runner, task_names_to_use=local_tasks, num_parallel_envs=num_parallel_envs, reset_type='ori')
    print(f'[RANK {rank}] Conducting rollout evaluation')
    rollout_results = env_runner.run(model.module, n_video=0, do_tqdm=train_cfg.use_tqdm) 
    print(f'[RANK {rank}] Rollout results: {rollout_results}')
    global_step = 0
    sync_rollout_results_via_file(rollout_results, logger, global_step)

    # Set up RL optimizer components
    if rank == 0:
        print('Setting up RL optimizer components')
    
    # Create rollout generator
    reward_func = instantiate(cfg.reward_function)

    rollout_generator = RolloutGenerator(
        rloo_batch_size=cfg.algo.rloo_batch_size,
        demo_batch_size=cfg.train_dataloader.batch_size // world_size,
        early_stop_percentage=cfg.algo.early_stop_porcentage,
        enable_dynamic_sampling=cfg.algo.enable_dynamic_sampling,
        task_names_to_use=local_tasks,
        env_runner=env_runner,
        create_env=True
    )
    
    # Create RL optimizer
    rl_optimizer = RLOptimizer(
        rollout_generator=rollout_generator,
        reward_function=reward_func,
        ppo_clip_range=cfg.algo.ppo_clip_range,
        num_ppo_epochs=cfg.algo.num_ppo_epochs,
        ppo_batch_size=cfg.algo.ppo_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        grad_norm_clip=train_cfg.grad_clip,
        use_minibatch_permutation=cfg.algo.use_minibatch_permutation,
        ppo_clip_high=cfg.algo.ppo_clip_high if hasattr(cfg.algo, 'ppo_clip_high') else None,
        use_token_level_loss_avg=cfg.algo.use_token_level_loss_avg if hasattr(cfg.algo, 'use_token_level_loss_avg') else False
    )
    
    # Synchronize all processes before training starts
    dist.barrier()
    
    # Training loop
    if rank == 0:
        print('Starting training loop')
    data_iter = iter(train_dataloader)
    t0 = time.time()
    epoch = 0
    
    for global_step in tqdm(range(total_steps), desc=f'Training with {world_size} GPUs'):
        model.train()

        try:
            data = next(data_iter)
        except StopIteration:
            # Reset the data iterator
            data_iter = iter(train_dataloader)
            data = next(data_iter)
        
        data = utils.map_tensor_to_device(data, device)
        
        # Run optimization using the standalone RL optimizer
        metrics = rl_optimizer.optimize(
            model_interface=model_adapter,
            batch=data,
            optimizers=optimizers,
            data_iterator=data_iter,
            dataloader=train_dataloader
        )
        
        print(f'rank {rank} metrics {metrics}')
        
        # Log metrics (only rank 0)
        info = {'epoch': epoch}
        info.update({
            "lr_0": optimizers[0].param_groups[0]['lr'],
            "lr_1": optimizers[1].param_groups[0]['lr'],
        })
        
        info = {cfg.logging_folder: info}

        for key, value in metrics.items():
            info[key] = value
        
        if rank == 0:
            logger.log(info, global_step)
        
        # Check if we should exit early
        if train_cfg.cut and global_step >= train_cfg.cut:
            break
        
        # Update epoch counter and learning rate
        if (global_step + 1) % steps_per_epoch == 0:
            epoch += 1
            t1 = time.time()
            print(f"[RANK {rank}] Epoch: {epoch:3d} | time: {(t1-t0)/60:4.2f}")
            t0 = time.time()
            [scheduler.step() for scheduler in schedulers]
        
        # Save model checkpoint (only rank 0)
        if rank == 0 and (global_step + 1) % save_interval_steps == 0:
            model_checkpoint_name_ep = os.path.join(
                    experiment_dir, f"multitask_model_step_{global_step:06d}.pth"
                )
            utils.save_state({'model': model}, model_checkpoint_name_ep)
        
        # Synchronize all processes
        dist.barrier()
        
        # Perform rollout evaluation if needed
        if cfg.rollout.enabled and (global_step + 1) % rollout_interval_steps == 0:
            print(f'[RANK {rank}] Conducting rollout evaluation')
            rollout_results = env_runner.run(model.module, n_video=0, do_tqdm=train_cfg.use_tqdm) 
            print(f'[RANK {rank}] Rollout results: {rollout_results}')
            sync_rollout_results_via_file(rollout_results, logger, global_step)
        
        # Synchronize all processes
        dist.barrier()
    
    # Clean up
    rollout_generator.cleanup()
    
    if rank == 0:
        print("[info] Finished training")
        wandb.finish()
        
    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()