MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
python train_ript_pi0.py \
  --config-name train_base_rl_openvla_oft \
  algo=pi0_cfg_rl \
  algo.norm_stats_path=/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json \
  algo.policy.pretrained_path=/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch \
  training.n_steps=20 \
  rollout.enabled=true \
  rollout.interval=10 \
  task=libero_spatial_rl \
  logging.mode=disabled