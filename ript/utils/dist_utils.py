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
import fcntl
import numpy as np
import torch
import torch.nn as nn
import ript.utils.utils as utils
from pyinstrument import Profiler
from ript.utils.logger import Logger, flatten_dict
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import multiprocessing
import json
import uuid
def write_json_atomic(filepath, data):
    tmp_filepath = filepath + ".tmp"
    with open(tmp_filepath, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp_filepath, filepath)

def read_json(filepath):
    with open(filepath, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data

def sync_rollout_results_via_file(rollout_results, logger, global_step):
    """
    Synchronize rollout results across processes using file-based communication.
    
    Args:
        rollout_results (dict): This rank's rollout results (no videos).
        logger: Logger object with a `.log(dict, step=global_step)` method.
        global_step (int): Step number for logging.
    
    Returns:
        None
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
      dist.barrier()

    if 'rollout_videos' in rollout_results:
        del rollout_results['rollout_videos']

    # --- Generate synchronized tmp_id ---
    if dist.is_initialized():
        if rank == 0:
            tmp_id = uuid.uuid4().hex
        else:
            tmp_id = None
        tmp_id_list = [tmp_id]
        dist.broadcast_object_list(tmp_id_list, src=0)
        tmp_id = tmp_id_list[0]
    else:
        tmp_id = uuid.uuid4().hex

    tmp_dir = f"/tmp/rollout_results_sync_{tmp_id}"
    os.makedirs(tmp_dir, exist_ok=True)
    rank_filepath = os.path.join(tmp_dir, f"rollout_rank_{rank}.json")

    # --- Write this rank's result to file ---
    write_json_atomic(rank_filepath, rollout_results)

    if dist.is_initialized():
        dist.barrier()

    # --- Aggregation by Rank 0 ---
    if rank == 0:
        print("[Rank 0] Waiting for all rollout results...")
        while True:
            files = [f for f in os.listdir(tmp_dir) if f.startswith("rollout_rank_") and f.endswith(".json")]
            if len(files) >= world_size:
                break
            time.sleep(1)

        print("[Rank 0] All rollout results received.")

        gathered_results = {'rollout': {}, 'rollout_success_rate': {}}
        overall_sr = 0
        overall_reward = 0
        num_solved = 0
        all_rollout_cnt = 0

        for f in files:
            result = read_json(os.path.join(tmp_dir, f))
            local_rollout_cnt = result['rollout']['rollout_count']
            all_rollout_cnt += local_rollout_cnt

            overall_sr += result['rollout']['overall_success_rate'] * local_rollout_cnt
            overall_reward += result['rollout']['overall_average_reward'] * local_rollout_cnt
            num_solved += result['rollout']['environments_solved']

            for env_name, env_success_rate in result['rollout_success_rate'].items():
                gathered_results['rollout_success_rate'][env_name] = env_success_rate

        gathered_results['rollout']['overall_success_rate'] = overall_sr / all_rollout_cnt
        gathered_results['rollout']['overall_average_reward'] = overall_reward / all_rollout_cnt
        gathered_results['rollout']['environments_solved'] = num_solved
        gathered_results['rollout']['rollout_count'] = all_rollout_cnt

        print(f"[Rank 0] gathered_results: {gathered_results}")
        print(f"[info]     success rate: {gathered_results['rollout']['overall_success_rate']:1.3f} \
                | environments solved: {gathered_results['rollout']['environments_solved']}")
        logger.log(gathered_results, step=global_step)

        # Optional: clean up
        for f in files:
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)
        return gathered_results
    
    else:
        return None

##############################
# Evaluation Worker Function
##############################
def eval_worker(rank, device_number, cfg, checkpoint_path, local_tasks, result_subfolder, global_step):
    """
    This function loads the model checkpoint, instantiates the env_runner,
    runs the rollout evaluation on the local tasks, and writes the result to a JSON file.
    """
    from hydra.utils import instantiate  # re-import in new process
    
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_number
    print('device_id', device_number)
    print(f'[Eval Worker][Rank {rank}] rank {rank} CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    print(f"[Eval Worker][Rank {rank}] Loading model from checkpoint {checkpoint_path} for evaluation at step {global_step}.")
    
    # Instantiate the model for evaluation
    try:
        model = instantiate(
            cfg.algo.policy,
            shape_meta=cfg.task.shape_meta,
            task_names_to_use=local_tasks,
            demo_batch_size=cfg.train_dataloader.batch_size // torch.cuda.device_count(),
            create_env=False
        )
    except Exception as e:
        print(f"[Eval Worker][Rank {rank}] Initializing the plain VLA model without RL")
        model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)

    model.to(device)
    model.eval()

    print(f"[Eval Worker][Rank {rank}] loading checkpoint {checkpoint_path}")
    
    # Load checkpoint and soft-load weights
    state_dict = utils.load_state(checkpoint_path)
    loaded_state_dict = state_dict['model']
    loaded_state_dict = {k[7:]: v for k, v in loaded_state_dict.items()}
    utils.soft_load_state_dict(model, loaded_state_dict)

    print(f"[Eval Worker][Rank {rank}] setting up env runner")
    
    # Instantiate the environment runner for rollout evaluation
    env_runner = instantiate(cfg.task.env_runner, task_names_to_use=local_tasks, num_parallel_envs=1)

    print(f"[Eval Worker][Rank {rank}] running rollout")
    rollout_results = env_runner.run(model, n_video=0, do_tqdm=cfg.training.use_tqdm)

    print(f"[Eval Worker][Rank {rank}] saving results to {result_subfolder}")
    # Write rollout results to a JSON file (one per GPU)
    result_file = os.path.join(result_subfolder, f"rollout_rank_{rank}_step_{global_step}.json")
    with open(result_file, "w") as f:
        json.dump(rollout_results, f)
    print(f"[Eval Worker][Rank {rank}] Evaluation complete, results saved to {result_file}.")

def read_json(filepath):
    with open(filepath, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data

##############################
# Evaluation Monitor Function
##############################
def eval_monitor(rank, device_number, cfg, experiment_dir, local_tasks, world_size):
    """
    This monitor process (spawned per GPU) continuously polls for a new rollout trigger.
    When a trigger is found, it spawns an evaluation worker to run the rollout on the local tasks.
    For rank==0, after its own evaluation, it waits until all GPUs finish and then aggregates the results and logs them.
    """
    trigger_dir = os.path.join(experiment_dir, "rollout_triggers")
    os.makedirs(trigger_dir, exist_ok=True)
    processed_triggers = set()
    
    print(f"[Eval Monitor][Rank {rank}] Starting monitor on GPU {device_number}.")
    while True:
        # Look for new trigger files (expected name: rollout_step_{global_step}.trigger)
        triggers = [f for f in os.listdir(trigger_dir) if f.endswith(".trigger")]
        triggers = sorted(triggers)

        # print(f"[Eval Monitor][Rank {rank}] trigger dir: {trigger_dir}, files: {os.listdir(trigger_dir)}")
        
        # Check for a stop signal (a file 'stop_monitor' in experiment_dir)
        if os.path.exists(os.path.join(experiment_dir, "stop_monitor")) and len(triggers) == len(processed_triggers):
            print(f"[Eval Monitor][Rank {rank}] Stop signal received. Exiting monitor.")
            break
    
        # print(f"[Eval Monitor][Rank {rank}] triggers: {triggers}")

        for trig in triggers:
            # print(f"[Eval Monitor][Rank {rank}] checking trigger {trig}")

            if trig in processed_triggers:
                # print(f"[Eval Monitor][Rank {rank}] trigger {trig} already processed")
                continue
            try:
                # Extract global_step from filename
                global_step = int(trig.split("_")[-1].split(".")[0])
            except Exception:
                continue
            
            checkpoint_path = os.path.join(experiment_dir, f"multitask_model_step_{global_step:06d}.pth")
            if not os.path.exists(checkpoint_path):
                print(f'[Eval Monitor][Rank {rank}] Waiting for checkpoint {checkpoint_path} to be available')
                continue  # wait until checkpoint is available
            
            # Create a unique result subfolder for this rollout trigger
            result_subfolder = os.path.join(experiment_dir, "rollout_results", f"rollout_step_{global_step:06d}")
            os.makedirs(result_subfolder, exist_ok=True)
            
            print(f"[Eval Monitor][Rank {rank}] Detected trigger {trig}. Spawning eval worker for step {global_step}.")
            # Spawn the evaluation worker process for this GPU
            multiprocessing.set_start_method('spawn')
            p = multiprocessing.Process(
                target=eval_worker,
                args=(rank, device_number, cfg, checkpoint_path, local_tasks, result_subfolder, global_step)
            )
            p.start()
            p.join()  # Wait for the evaluation worker to finish
            processed_triggers.add(trig)
            print(f"[Eval Monitor][Rank {rank}] processed trigger {trig}")
            # Only the monitor on rank 0 aggregates the results.
            if rank == 0:
                print(f"[Aggregator][Rank 0] Waiting for evaluation results for step {global_step} from all GPUs.")
                # Wait until result_subfolder has one file per GPU.
                timeout = 3600 * 4  # seconds; adjust as needed; 4-hours is a good default
                start_time = time.time()
                while True:
                    files = [f for f in os.listdir(result_subfolder) if f.endswith(".json")]
                    time_elapsed = time.time() - start_time
                    if len(files) >= world_size:
                        break
                    if time_elapsed > timeout:
                        print(f"[Aggregator][Rank 0] Timeout waiting for all results at step {global_step}.")
                        break
                    time.sleep(2)
                    print(f"[Aggregator][Rank 0] Waiting for all results at step {global_step}: {len(files)}/{world_size}.")
                    print(f"[Aggregator][Rank 0] files: {files}")
                
                print("[Rank 0] All rollout results received.")

                gathered_results = {'rollout': {}, 'rollout_success_rate': {}}
                overall_sr = 0
                overall_reward = 0
                num_solved = 0
                all_rollout_cnt = 0

                for f in files:
                    result = read_json(os.path.join(result_subfolder, f))
                    local_rollout_cnt = result['rollout']['rollout_count']
                    all_rollout_cnt += local_rollout_cnt

                    overall_sr += result['rollout']['overall_success_rate'] * local_rollout_cnt
                    overall_reward += result['rollout']['overall_average_reward'] * local_rollout_cnt
                    num_solved += result['rollout']['environments_solved']

                    for env_name, env_success_rate in result['rollout_success_rate'].items():
                        gathered_results['rollout_success_rate'][env_name] = env_success_rate

                gathered_results['rollout']['overall_success_rate'] = overall_sr / all_rollout_cnt
                gathered_results['rollout']['overall_average_reward'] = overall_reward / all_rollout_cnt
                gathered_results['rollout']['environments_solved'] = num_solved

                print(f"[Rank 0] gathered_results: {gathered_results}")
                print(f"[info]     success rate: {gathered_results['rollout']['overall_success_rate']:1.3f} \
                        | environments solved: {gathered_results['rollout']['environments_solved']}")
                agg_result_file = os.path.join(experiment_dir, f"aggregated_rollout_results_{global_step:06d}.json")
                with open(agg_result_file, "w") as f:
                    json.dump({"global_step": global_step, "rollout_results": gathered_results}, f)
                print(f"[Aggregator][Rank 0] Aggregated results dumped to {agg_result_file}.")

        # print(f"[Eval Monitor][Rank {rank}] Sleeping for 5 seconds")
        time.sleep(5)  # Poll every 5 seconds

def log_aggregated_rollout_results(experiment_dir, global_step, logger):
    # Look for any aggregated rollout result files in the experiment directory.
    agg_files = [
        f for f in os.listdir(experiment_dir)
        if f.startswith("aggregated_rollout_results_") and f.endswith(".json")
    ]
    agg_files = sorted(agg_files)
    for agg_file in agg_files:
        agg_result_file = os.path.join(experiment_dir, agg_file)
        with open(agg_result_file, "r") as f:
            agg_results = json.load(f)
        # Use the global_step stored inside the aggregated result file for logging.
        logged_step = agg_results["global_step"]
        flat_results = flatten_dict(agg_results["rollout_results"])
        for key, value in flat_results.items():
            try:
                wandb.define_metric(key, step_metric='rollout_step')
            except Exception:
                pass
            wandb.log({key: value, 'rollout_step': logged_step})

        print(f"[Aggregator][Rank 0] Logged rollout results for step {logged_step} from {agg_result_file}")
        os.remove(agg_result_file)

