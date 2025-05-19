MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=$1\
  --master_port $MASTER_PORT train_ript_openvla_oft.py\
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml\
  exp_name=OpenVLA-OFT_libero_spatial_train\
  variant_name=bz_24_scale_2.0\
  algo.model_seed=1\
  train_dataloader.batch_size=24\
  training.n_steps=12\
  training.save_interval=1 \
  algo.env_runner.num_parallel_envs=2\
  algo.model_seed=0 \
  algo.scale_factor=2.0 \
  algo.checkpoint_path=PUT_SFT_CHECKPOINT_FROM_OFFICIAL_REPO_HERE\
  algo.header_checkpoint=PUT_SFT_SCALE_HEADER_CHECKPOINT_FROM_MODEL_ZOO_HERE\
  algo.lora_adaptor_ckpt=PUT_RIPT_LORA_ADAPTOR_CHECKPOINT_FROM_MODEL_ZOO_HERE # If RIPT directly from SFT, set to null
