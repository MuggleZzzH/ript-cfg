MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=$1\
  --master_port $MASTER_PORT train_ript_openvla_oft.py\
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml\
  exp_name=OpenVLA-OFT_LIBERO_SPATIAL_eval\
  variant_name=base_bz_24\
  algo.model_seed=1\
  algo.eval_only=True\
  algo.checkpoint_path=PUT_SFT_CHECKPOINT_FROM_OFFICIAL_REPO_HERE\
  algo.header_checkpoint=PUT_SFT_SCALE_HEADER_CHECKPOINT_FROM_MODEL_ZOO_HERE\
  algo.lora_adaptor_ckpt=PUT_RIPT_LORA_ADAPTOR_CHECKPOINT_FROM_MODEL_ZOO_HERE
