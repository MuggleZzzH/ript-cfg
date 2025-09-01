#!/bin/bash

# Train autoencoder for LIBERO Spatial tasks
python train_quest_sft.py --config-name=train_autoencoder_libero_spatial.yaml \
    task=libero_spatial \
    algo=quest \
    exp_name=final \
    variant_name=spatial_block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=0 