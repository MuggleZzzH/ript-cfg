import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import warnings

import random
import numpy as np
import torch
import torch.nn as nn
import ript.utils.utils as utils
from pyinstrument import Profiler
from ript.utils.logger import Logger
import gc

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    start_epoch, steps, wandb_id = 0, 0, None
    if train_cfg.auto_continue:
        checkpoint_path = experiment_dir.rsplit('/', 1)[0] + f'/stage_{cfg.stage - 1}'
        if 'libero' in checkpoint_path and cfg.stage == 2:
            checkpoint_path = checkpoint_path.replace('10', '90') # since we want to initialize the model from the libero_90 benchmark
    elif train_cfg.resume and len(os.listdir(experiment_dir)) > 0: 
        checkpoint_path = experiment_dir
    else: 
        checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f'loading from checkpoint {checkpoint_path}')
        state_dict = utils.load_state(checkpoint_path)
        loaded_state_dict = state_dict['model']
        
        # Below line allows loading state dicts with some mismatched parameters
        utils.soft_load_state_dict(model, loaded_state_dict)
    else:
        print('starting from scratch')

    dataset = instantiate(cfg.task.dataset)
    model.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)


    if cfg.rollout.enabled:
        env_runner = instantiate(cfg.task.env_runner)
    print('Saving to:', experiment_dir)
    print('Experiment name:', experiment_name)

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        # id=wandb_id,
        **cfg.logging
    )

    logger = Logger(train_cfg.log_interval)

    print('Training...')

    best_rollout_sr = 0.0
    to_save_best_model = False

    for epoch in range(start_epoch, train_cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        if train_cfg.do_profile:
            profiler = Profiler()
            profiler.start()
        
        gradient_accumulation_steps = train_cfg.gradient_accumulation_steps
        gradient_step = 0
        last_grad_norm = 0.0

        for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
            data = utils.map_tensor_to_device(data, device)
            
            if gradient_step == 0:
                # print('zero grad')
                for optimizer in optimizers:
                    optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(False):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_cfg.use_amp):
                    loss, info = model.compute_loss(data)
            
                scaler.scale(loss).backward()

            gradient_step += 1

            # print('gradient step', gradient_step)

            if gradient_step == gradient_accumulation_steps:
                # print('optimizer step')
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                if train_cfg.grad_clip is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg.grad_clip
                    )

                for optimizer in optimizers:
                    scaler.step(optimizer)
                
                scaler.update()

                gradient_step = 0

            info.update({
                'epoch': epoch
            })
            if train_cfg.grad_clip is not None:
                if gradient_step == 0:
                    last_grad_norm = grad_norm.item()
                info.update({
                    "grad_norm": last_grad_norm,
                })  
            info.update({
                "lr_0": optimizers[0].param_groups[0]['lr'],
                "lr_1": optimizers[1].param_groups[0]['lr'],
            })
            info = {cfg.logging_folder: info}
            training_loss += loss.item()
            steps += 1
            logger.update(info, steps)

            if train_cfg.cut and idx > train_cfg.cut:
                break
        
            # import pdb; pdb.set_trace()

        if train_cfg.do_profile:
            profiler.stop()
            profiler.print()

        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
        )

        if cfg.rollout.enabled and epoch > 0 and epoch % cfg.rollout.interval == 0:
            rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=train_cfg.use_tqdm)
            print(
                f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
                    | environments solved: {rollout_results['rollout']['environments_solved']}")
            logger.log(rollout_results, step=steps)

            if rollout_results['rollout']['overall_success_rate'] > best_rollout_sr:
                best_rollout_sr = rollout_results['rollout']['overall_success_rate']
                to_save_best_model = True
        
        if (epoch % train_cfg.save_interval == 0 and epoch > 0) or to_save_best_model:
            if to_save_best_model:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_best_sr_{best_rollout_sr:1.3f}.pth"
                    )
                to_save_best_model = False
            elif cfg.training.save_all_checkpoints:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth"
                    )
            else:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_latest.pth"
                    )
            utils.save_state({'model': model}, model_checkpoint_name_ep)
        
        
        [scheduler.step() for scheduler in schedulers]
    print("[info] finished learning\n")
    wandb.finish()

if __name__ == "__main__":
    main()