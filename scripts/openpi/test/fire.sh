python train_ript_pi0.py \
  --config-name train_base_rl_openvla_oft \
  algo=pi0_cfg_rl \
  algo.norm_stats_path=/abs/to/norm_stats.json \
  algo.policy.pretrained_path=/abs/to/pi0_checkpoint_dir \
  training.n_steps=20 \
  rollout.enabled=true \
  rollout.interval=10 \
  task=libero_spatial_rl