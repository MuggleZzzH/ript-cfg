NCCL_TIMEOUT=108000 torchrun --nproc_per_node=$1 --master_port 12345 eval_quest.py --config-name=train_prior_rl.yaml \
  task=libero_long_rl \
  exp_name=LIBERO_LONG \
  variant_name=eval_only\
  rollout.rollouts_per_env=48 \
  task.env_runner.num_parallel_envs=8\
  checkpoint_path=FILL_IN_YOUR_CHECKPOINT_PATH_FROM_SFR_OR_RIPT \
  task.env_runner.reset_type=ori \
  seed=0
