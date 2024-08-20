#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

cd 6DoF

# Debugging
BS_PER_GPU=1
N_ACCUMULATION=1

CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch \
    --config_file ../configs/deepspeed_debug.yaml \
    train_eschernet_deepspeed.py \
    --train_data_dir /data2/wlsgur4011/zero123_data_small/views_release \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_batch_size ${BS_PER_GPU} \
    --dataloader_num_workers 16 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps ${N_ACCUMULATION} \
    --T_in 3 \
    --T_out 3 \
    --T_in_val 10 \
    --validation_steps 6 \
    --checkpointing_steps 10 \
    --num_validation_batches 1 \
    --report_to tensorboard \
    --output_dir logs_debug_node${node}_deepspeed \
    --resume_from_checkpoint latest \
    --tracker_project_name eschernet

# ... 


















set +x; duration=SECONDS; RED='\033[0;31m'; Yellow='\033[1;33m'; Green='\033[0;32m'; NC='\033[0m'; echo -e "RED$((duration / 3600))hNC Yellow$((duration / 60 % 60))mNC Green$((duration % 60))sNC elapsed."