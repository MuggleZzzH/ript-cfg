MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=$1\
  --master_port $MASTER_PORT train_ript_openvla_oft.py\
  --config-name=train_rl_openvla_oft_all_task_long.yaml\
  exp_name=OpenVLA-OFT_libero_long_train\
  variant_name=bz_24_scale_5.0\
  algo.model_seed=1\
  train_dataloader.batch_size=24\
  training.n_steps=12\
  training.save_interval=1 \
  algo.env_runner.num_parallel_envs=2\
  algo.model_seed=0 \
  algo.scale_factor=5.0 \
  algo.rollout_training_task_names=[LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket,LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate,LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate,LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket] \
  algo.checkpoint_path=PUT_SFT_CHECKPOINT_FROM_OFFICIAL_REPO_HERE\
  algo.header_checkpoint=PUT_SFT_SCALE_HEADER_CHECKPOINT_FROM_MODEL_ZOO_HERE\
  algo.lora_adaptor_ckpt=PUT_RIPT_LORA_ADAPTOR_CHECKPOINT_FROM_MODEL_ZOO_HERE # If RIPT directly from SFT, set to null
