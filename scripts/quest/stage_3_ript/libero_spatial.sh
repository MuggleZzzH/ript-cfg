MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=$1 train_ript.py  --config-name=train_prior_rl.yaml \
  task=libero_spatial_rl \
  exp_name=quest_rl_spatial \
  variant_name=LIBERO_SPATIAL_ppo_epoch_20_ppo_batch_6_rloo_8_demo_batch_180_stop_0.8_dynamic_sampling\
  train_dataloader.batch_size=180 \
  algo.rloo_batch_size=8 \
  algo.ppo_batch_size=2 \
  algo.num_ppo_epochs=20 \
  rollout.rollouts_per_env=48 \
  task.env_runner.num_parallel_envs=4\
  training.n_steps=30\
  training.rollout_steps=1\
  algo.early_stop_porcentage=0.8\
  algo.enable_dynamic_sampling=true \
  algo.use_token_level_loss_avg=true \
  checkpoint_path=FILL_IN_YOUR_CHECKPOINT_PATH_FROM_SFR_OR_RIPT \
  algo.policy.scheduler_factory.eta_min=9e-7 \
  task.env_runner.reset_type=ori