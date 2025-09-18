import os
# 设置NCCL通信超时时间，单位为秒（这里设置为30小时，防止大规模分布式训练时超时）
os.environ["NCCL_TIMEOUT"] = "108000"
# 设置TensorFlow的日志级别，2表示只显示warning和error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import time
import hydra  # 配置管理库
import wandb  # Weights & Biases，用于实验追踪和可视化
import datetime
from hydra.utils import instantiate  # hydra的实例化工具
from omegaconf import OmegaConf  # 配置对象
from tqdm import tqdm  # 进度条显示

import torch
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练相关
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data.dataloader import default_collate  # 默认的batch合并函数
import ript.utils.utils as utils  # 工具函数
from ript.utils.logger import Logger  # 日志记录器
from ript.utils.dist_utils import sync_rollout_results_via_file  # 分布式结果同步工具

# 注册新的OmegaConf解析器，支持在配置文件中动态eval表达式
OmegaConf.register_new_resolver("eval", eval, replace=True)

def collate_fn_state(batch):
    # 处理特殊字段'init_state'，对状态进行padding和mask处理
    states = [item['init_state']['states'] for item in batch]
    max_len = max(s.shape[-1] for s in states)  # 找到最长的状态长度

    padded_states = []  # 存储补齐后的状态
    masks = []          # 存储对应的mask
    modified_batch = [] # 存储去除'init_state'后的其他字段

    for item in batch:
        # 对每个状态进行补齐
        tensor = torch.as_tensor(item['init_state']['states']).float()
        pad_size = max_len - tensor.shape[-1]
        padded = torch.nn.functional.pad(tensor, (0, pad_size))
        padded_states.append(padded)

        # 生成mask，原始长度为True，补齐部分为False
        mask = torch.ones(tensor.shape[-1], dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
        masks.append(mask)

        # 去除'init_state'字段，保留其他字段
        modified_item = {key: item[key] for key in item.keys() if key != 'init_state'}
        modified_batch.append(modified_item)

    # 对其他字段使用默认的collate方式合并
    collated_batch = default_collate(modified_batch)

    # 将处理好的状态和mask重新加入batch
    collated_batch['init_state'] = {}
    collated_batch['init_state']['states'] = torch.stack(padded_states)
    collated_batch['init_state']['pad_mask'] = torch.stack(masks)

    return collated_batch

@hydra.main(config_path="config", version_base=None)
def main(cfg):
    # 设置随机种子，保证实验可复现
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # 初始化分布式训练环境，使用nccl后端，设置超时时间
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    # 获取分布式相关信息
    rank = dist.get_rank()  # 当前进程编号
    world_size = dist.get_world_size()  # 总进程数
    local_rank_env = os.environ.get('LOCAL_RANK')
    device_id = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)  # 设置当前进程使用的GPU

    # 打印CUDA可见设备信息，便于调试
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print('CUDA_VISIBLE_DEVICES:', cuda_visible_devices.split(',') if cuda_visible_devices else [''])
    device_number = cuda_visible_devices.split(',')[device_id] if cuda_visible_devices else str(device_id)
    if cuda_visible_devices and local_rank_env is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_number
        print('device_id', device_id)
        print(f'rank {rank} CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    else:
        print('device_id', device_id)
        print(f'rank {rank} using device {device}')

    # 按GPU划分任务
    all_tasks = cfg.task.task_names_to_use
    if all_tasks is None:
        # 如果未指定任务，则从benchmark中获取全部任务
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(cfg.task.benchmark_name.lower())()
        all_tasks = benchmark.get_task_names()
        print('using all tasks from benchmark', benchmark.name)
    # 将任务均匀分配到每个rank
    rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
    for task_i, task_name in enumerate(all_tasks):
        rank_to_tasks[task_i % world_size].append(task_name)
    local_eval_tasks = rank_to_tasks[rank]  # 当前rank负责的评估任务

    # 训练任务分配（可与评估任务不同）
    if cfg.algo.rollout_training_task_names is not None:
        all_train_tasks = cfg.algo.rollout_training_task_names
        rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
        for task_i, task_name in enumerate(all_train_tasks):
            rank_to_tasks[task_i % world_size].append(task_name)
        local_train_tasks = rank_to_tasks[rank]
    else:
        local_train_tasks = local_eval_tasks

    # 打印每个rank的任务分配信息
    print(f'[RANK {rank}] World size: {world_size}, Device: {device}, Tasks: {local_train_tasks}\n')

    # 计算每轮训练的样本数、步数等
    total_examples = cfg.task.rollouts_per_env * len(all_tasks)
    steps_per_epoch = total_examples // cfg.train_dataloader.batch_size
    save_interval_steps = train_cfg.save_interval

    # 总训练步数
    if train_cfg.n_steps != -1:
        total_steps = train_cfg.n_steps
    else:
        total_steps = steps_per_epoch * train_cfg.n_epochs

    # rollout评估的间隔步数
    if train_cfg.rollout_steps != -1:
        rollout_interval_steps = train_cfg.rollout_steps
    else:
        rollout_interval_steps = steps_per_epoch * cfg.rollout.interval

    # rank 0 打印训练配置信息
    if rank == 0:
        print('Training Configuration:')
        print(f"\tTotal Examples per Epoch: {total_examples}")
        print(f"\tSteps per Epoch: {steps_per_epoch}")
        print(f"\tTotal Steps: {total_steps}")
        print(f"\tSave Interval Steps: {save_interval_steps}")
        print(f"\tRollout Interval Steps: {rollout_interval_steps}")

    # 加载模型，OpenVLA策略会自动处理DDP封装
    model = instantiate(cfg.algo.policy, device_id=device_id)

    # 获取实验目录和实验名
    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    if rank == 0:
        print('Saving to:', experiment_dir)
        print('Experiment name:', experiment_name)

        import os as _os
        backend = str(_os.environ.get('LOG_BACKEND', 'wandb')).lower()
        logger = Logger(train_cfg.log_interval, backend=backend)
        try:
            if backend == 'swanlab':
                import swanlab as _lb
                _mode = str(_os.environ.get('SWANLAB_MODE', '')).lower()
                _lb.init(
                    project=str(getattr(cfg, 'logging_folder', 'ript')),
                    experiment_name=str(experiment_name),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    mode=('offline' if _mode == 'offline' else 'online'),
                )
            elif backend == 'wandb':
                import wandb as _lb
                _lb.init(
                    dir=experiment_dir,
                    name=experiment_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
        except Exception as _e:
            print(f"[RANK {rank}] Warning: logger backend init failed: {getattr(_e, 'args', _e)}")
    else:
        logger = None

    # 如果不是只做评估，则进行训练相关初始化
    if not cfg.algo.eval_only:
        # 创建模型和header的优化器
        optimizer_factory_model = instantiate(cfg.algo.optimizer_factory)
        optimizer_factory_header = instantiate(cfg.algo.optimizer_factory, lr = cfg.algo.header_lr)
        optimizers = [optimizer_factory_model(model.trainable_params['model']), optimizer_factory_header(model.trainable_params['header'])]

        # 创建学习率调度器
        scheduler_factory = instantiate(cfg.algo.scheduler_factory)
        schedulers = [scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

        # 构建数据集和dataloader
        dataset = instantiate(cfg.task.dataset, task_names_to_use=local_train_tasks)

        train_dataloader = instantiate(
            cfg.train_dataloader,
            dataset=dataset,
            collate_fn=collate_fn_state,
            batch_size=cfg.train_dataloader.batch_size // world_size
        )

        # 环境并行数设置
        num_parallel_envs = cfg.algo.num_parallel_envs
        if len(local_train_tasks) > 1:
            num_parallel_envs = 1
        env_runner = instantiate(cfg.algo.env_runner, task_names_to_use=local_train_tasks, num_parallel_envs=num_parallel_envs)

        # RL优化器相关组件初始化
        if rank == 0:
            print('Setting up RL optimizer components')

        # 创建奖励函数
        reward_function = instantiate(cfg.reward_function)

        # 创建rollout生成器
        rollout_generator_factory = instantiate(cfg.algo.rollout_generator_factory)
        rollout_generator = rollout_generator_factory(
            env_runner=env_runner,
            task_names_to_use=local_train_tasks,
            demo_batch_size=cfg.train_dataloader.batch_size // world_size
        )

        # 创建RL优化器
        rl_optimizer_factory = instantiate(cfg.algo.rl_optimizer_factory, enable_rollout_stats_tracking=True)
        rl_optimizer = rl_optimizer_factory(
            rollout_generator=rollout_generator,
            reward_function=reward_function
        )

    # 如果需要rollout评估，初始化评估环境
    if cfg.rollout.enabled:
        num_parallel_envs = cfg.algo.env_runner.num_parallel_envs
        print(f'Eval env num_parallel_envs: {num_parallel_envs}')
        eval_env_runner = instantiate(cfg.algo.env_runner, task_names_to_use=local_eval_tasks, num_parallel_envs=num_parallel_envs, max_episode_length=None, use_laplace_sampling=False, scale_factor=1.0)
        if cfg.algo.eval_only:
            model.model.eval()
            try:
                n_video = int(getattr(cfg.rollout, 'n_video', int(os.environ.get('PI0_N_VIDEO', 0))))
            except Exception:
                n_video = 0
            rollout_results = eval_env_runner.run(model, n_video=n_video, do_tqdm=train_cfg.use_tqdm)
            print(f'[RANK {rank}] Rollout results: {rollout_results}')
            sync_rollout_results_via_file(rollout_results, logger, 0)

    # 所有进程同步，确保训练前准备工作完成
    dist.barrier()

    # 训练主循环
    if rank == 0:
        print('Starting training loop')
    data_iter = iter(train_dataloader)  # 获取数据迭代器
    t0 = time.time()  # 记录起始时间
    epoch = 0         # 记录当前epoch

    for global_step in tqdm(range(total_steps), desc=f'Training with {world_size} GPUs'):
        if cfg.algo.eval_only:
            break

        model.model.train()  # 设置模型为训练模式

        try:
            data = next(data_iter)
        except StopIteration:
            # 数据集遍历完，重置迭代器
            data_iter = iter(train_dataloader)
            data = next(data_iter)

        # 将数据移动到指定设备
        data = utils.map_tensor_to_device(data, device)

        # 使用RL优化器进行一次优化
        # 直接传递model，无需手动处理DDP
        metrics = rl_optimizer.optimize(
            model=model,
            batch=data,
            optimizers=optimizers,
            data_iterator=data_iter,
            dataloader=train_dataloader
        )

        print(f'rank {rank} metrics {metrics}')

        # 只在rank 0记录日志
        info = {'epoch': epoch}
        info.update({
            "lr_model": optimizers[0].param_groups[0]['lr'],
            "lr_header": optimizers[1].param_groups[0]['lr'],
        })

        info = {cfg.logging_folder: info}

        # 将metrics加入日志
        for key, value in metrics.items():
            info[key] = value

        if rank == 0 and logger is not None:
            try:
                logger.log(info, global_step)
            except Exception:
                pass

        # 判断是否提前终止训练
        if train_cfg.cut and global_step >= train_cfg.cut:
            break

        # 每个epoch结束时，更新epoch计数和学习率
        if global_step > 0 and global_step % steps_per_epoch == 0:
            epoch += 1
            t1 = time.time()
            print(f"[RANK {rank}] Epoch: {epoch:3d} | time: {(t1-t0)/60:4.2f}")
            t0 = time.time()
            for scheduler in schedulers:
                scheduler.step()

        # 按间隔保存模型权重
        if rank == 0 and global_step > 0 and global_step % save_interval_steps == 0:
            model_checkpoint_dir = os.path.join(experiment_dir, f"openvla_lora_step_{global_step:06d}")
            if hasattr(model.model, 'module'):
                # 如果模型被DDP封装
                peft_model = model.model.module
                action_head_model = model.action_head.module
                scale_head_model = model.scale_head.module
            else:
                # 未封装
                peft_model = model.model
                action_head_model = model.action_head
                scale_head_model = model.scale_head

            # 只保存LoRA适配器权重
            peft_model.save_pretrained(model_checkpoint_dir)
            headers_path = os.path.join(model_checkpoint_dir, "openvla_headers.pt")
            header_states = {
                'action_header': action_head_model.state_dict(),
                'scale_header': scale_head_model.state_dict()
            }
            torch.save(header_states, headers_path)

            print(f"[RANK {rank}] Saved LoRA weights and headers to {model_checkpoint_dir}")

        # 所有进程同步，保证保存和评估时一致
        dist.barrier()

        # 按间隔进行rollout评估
        if cfg.rollout.enabled and ((global_step % rollout_interval_steps == 0) or global_step == total_steps - 1 or rollout_interval_steps == 1):
            print(f'[RANK {rank}] Conducting rollout evaluation')
            model.model.eval()
            try:
                n_video = int(getattr(cfg.rollout, 'n_video', int(os.environ.get('PI0_N_VIDEO', 0))))
            except Exception:
                n_video = 0
            rollout_results = eval_env_runner.run(model, n_video=n_video, do_tqdm=train_cfg.use_tqdm)
            print(f'[RANK {rank}] Rollout results: {rollout_results}')

            # 分布式同步并记录rollout结果
            sync_rollout_results_via_file(rollout_results, logger, global_step)

        # 再次同步，保证所有进程步调一致
        dist.barrier()

    # 训练结束，清理rollout生成器
    rollout_generator.cleanup()

    if rank == 0:
        print("[info] Finished training")
        import os as _os
        backend = str(_os.environ.get('LOG_BACKEND', 'wandb')).lower()
        try:
            if backend == 'swanlab':
                import swanlab as _lb
                _lb.finish()
            elif backend == 'wandb':
                import wandb as _lb
                _lb.finish()
        except Exception:
            pass

    # 销毁分布式进程组，释放资源
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 
