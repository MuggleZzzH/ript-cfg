import os
os.environ["NCCL_TIMEOUT"] = "108000"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime
import time

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import default_collate

import ript.utils.utils as utils
from ript.utils.logger import Logger
from ript.utils.dist_utils import sync_rollout_results_via_file


def collate_fn_state(batch):
    states = [item['init_state']['states'] for item in batch]
    max_len = max(s.shape[-1] for s in states)

    padded_states = []
    masks = []
    modified_batch = []

    for item in batch:
        tensor = torch.as_tensor(item['init_state']['states']).float()
        pad_size = max_len - tensor.shape[-1]
        padded = torch.nn.functional.pad(tensor, (0, pad_size))
        padded_states.append(padded)

        mask = torch.ones(tensor.shape[-1], dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
        masks.append(mask)

        modified_item = {key: item[key] for key in item.keys() if key != 'init_state'}
        modified_batch.append(modified_item)

    collated_batch = default_collate(modified_batch)

    collated_batch['init_state'] = {}
    collated_batch['init_state']['states'] = torch.stack(padded_states)
    collated_batch['init_state']['pad_mask'] = torch.stack(masks)

    return collated_batch


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

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

    all_tasks = cfg.task.task_names_to_use
    if all_tasks is None:
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(cfg.task.benchmark_name.lower())()
        all_tasks = benchmark.get_task_names()
        print('using all tasks from benchmark', benchmark.name)
    rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
    for task_i, task_name in enumerate(all_tasks):
        rank_to_tasks[task_i % world_size].append(task_name)
    local_eval_tasks = rank_to_tasks[rank]

    if cfg.algo.rollout_training_task_names is not None:
        all_train_tasks = cfg.algo.rollout_training_task_names
        rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
        for task_i, task_name in enumerate(all_train_tasks):
            rank_to_tasks[task_i % world_size].append(task_name)
        local_train_tasks = rank_to_tasks[rank]
    else:
        local_train_tasks = local_eval_tasks

    print(f'[RANK {rank}] World size: {world_size}, Device: {device}, Tasks: {local_train_tasks}\n')

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
        rollout_interval_steps = steps_per_epoch * cfg.rollout.interval

    if rank == 0:
        print('Training Configuration:')
        print(f"\tTotal Examples per Epoch: {total_examples}")
        print(f"\tSteps per Epoch: {steps_per_epoch}")
        print(f"\tTotal Steps: {total_steps}")
        print(f"\tSave Interval Steps: {save_interval_steps}")
        print(f"\tRollout Interval Steps: {rollout_interval_steps}")

    model = instantiate(cfg.algo.policy, device_id=device_id)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    if rank == 0:
        print('Saving to:', experiment_dir)
        print('Experiment name:', experiment_name)

        import wandb
        wandb.init(
            dir=experiment_dir,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        logger = Logger(train_cfg.log_interval)
    else:
        logger = None

    if not cfg.algo.eval_only:
        optimizer_factory = instantiate(cfg.algo.optimizer_factory)
        optimizers = []
        # model group
        if hasattr(model, 'trainable_params') and 'model' in model.trainable_params and len(model.trainable_params['model']) > 0:
            optimizers.append(optimizer_factory(model.trainable_params['model']))
        # optional header group
        if hasattr(model, 'trainable_params') and 'header' in model.trainable_params and len(model.trainable_params['header']) > 0:
            opt_header = instantiate(cfg.algo.optimizer_factory, lr=getattr(cfg.algo, 'header_lr', cfg.algo.lr))
            optimizers.append(opt_header(model.trainable_params['header']))

        scheduler_factory = instantiate(cfg.algo.scheduler_factory)
        schedulers = [scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

        dataset = instantiate(cfg.task.dataset, task_names_to_use=local_train_tasks)
        train_dataloader = instantiate(
            cfg.train_dataloader,
            dataset=dataset,
            collate_fn=collate_fn_state,
            batch_size=cfg.train_dataloader.batch_size // world_size
        )

        num_parallel_envs = cfg.algo.env_runner.num_parallel_envs if hasattr(cfg.algo.env_runner, 'num_parallel_envs') else cfg.algo.num_parallel_envs
        if len(local_train_tasks) > 1:
            num_parallel_envs = 1
        env_runner = instantiate(cfg.algo.env_runner, task_names_to_use=local_train_tasks, num_parallel_envs=num_parallel_envs)

        if rank == 0:
            print('Setting up RL optimizer components')

        reward_function = instantiate(cfg.reward_function)
        rollout_generator_factory = instantiate(cfg.algo.rollout_generator_factory)
        rollout_generator = rollout_generator_factory(
            env_runner=env_runner,
            task_names_to_use=local_train_tasks,
            demo_batch_size=cfg.train_dataloader.batch_size // world_size
        )
        rl_optimizer_factory = instantiate(cfg.algo.rl_optimizer_factory, enable_rollout_stats_tracking=True)
        rl_optimizer = rl_optimizer_factory(
            rollout_generator=rollout_generator,
            reward_function=reward_function
        )

    if cfg.rollout.enabled:
        num_parallel_envs = cfg.algo.env_runner.num_parallel_envs if hasattr(cfg.algo.env_runner, 'num_parallel_envs') else cfg.algo.num_parallel_envs
        print(f'Eval env num_parallel_envs: {num_parallel_envs}')
        eval_env_runner = instantiate(cfg.algo.env_runner, task_names_to_use=local_eval_tasks, num_parallel_envs=num_parallel_envs, max_episode_length=None)
        if cfg.algo.eval_only:
            # Switch to eval if model exposes inner module
            if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                model.model.eval()
            rollout_results = eval_env_runner.run(model, n_video=0, do_tqdm=train_cfg.use_tqdm)
            print(f'[RANK {rank}] Rollout results: {rollout_results}')
            sync_rollout_results_via_file(rollout_results, logger, 0)

    dist.barrier()

    if rank == 0:
        print('Starting training loop')
    data_iter = iter(train_dataloader) if not cfg.algo.eval_only else None
    t0 = time.time()
    epoch = 0

    for global_step in tqdm(range(total_steps), desc=f'Training with {world_size} GPUs'):
        if cfg.algo.eval_only:
            break

        # optional train mode switch
        if hasattr(model, 'model') and hasattr(model.model, 'train'):
            model.model.train()

        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            data = next(data_iter)

        data = utils.map_tensor_to_device(data, device)

        metrics = rl_optimizer.optimize(
            model=model,
            batch=data,
            optimizers=optimizers,
            data_iterator=data_iter,
            dataloader=train_dataloader
        )

        print(f'rank {rank} metrics {metrics}')

        info = {'epoch': epoch}
        # log learning rates if available
        for i, optimizer in enumerate(optimizers):
            info[f"lr_{i}"] = optimizer.param_groups[0]['lr']
        info = {cfg.logging_folder: info}
        for key, value in metrics.items():
            info[key] = value

        if rank == 0:
            logger.log(info, global_step)

        if train_cfg.cut and global_step >= train_cfg.cut:
            break

        if global_step > 0 and global_step % steps_per_epoch == 0:
            epoch += 1
            t1 = time.time()
            print(f"[RANK {rank}] Epoch: {epoch:3d} | time: {(t1-t0)/60:4.2f}")
            t0 = time.time()
            for scheduler in schedulers:
                scheduler.step()

        if rank == 0 and global_step > 0 and global_step % save_interval_steps == 0:
            # Generic model save (best-effort)
            ckpt_dir = os.path.join(experiment_dir, f"pi0_step_{global_step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                # If model exposes inner module with state_dict
                target = model.model if hasattr(model, 'model') else model
                torch.save(target.state_dict(), os.path.join(ckpt_dir, 'pi0_policy.pt'))
                print(f"[RANK {rank}] Saved PI0 weights to {ckpt_dir}")
            except Exception as e:
                print(f"[RANK {rank}] Warning: failed to save PI0 weights: {e}")

        dist.barrier()

        if cfg.rollout.enabled and ((global_step % rollout_interval_steps == 0) or global_step == total_steps - 1 or rollout_interval_steps == 1):
            print(f'[RANK {rank}] Conducting rollout evaluation')
            if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                model.model.eval()
            rollout_results = eval_env_runner.run(model, n_video=0, do_tqdm=train_cfg.use_tqdm)
            print(f'[RANK {rank}] Rollout results: {rollout_results}')
            sync_rollout_results_via_file(rollout_results, logger, global_step)

        dist.barrier()

    rollout_generator.cleanup()

    if rank == 0:
        print("[info] Finished training")
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


