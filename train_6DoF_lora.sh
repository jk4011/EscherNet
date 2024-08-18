#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

cd 6DoF

N_GPUS=6
TOTAL_BS=672
N_ACCUMULATION=4

let BS_PER_GPU=${TOTAL_BS}/${N_GPUS}/${N_ACCUMULATION}
if [ $((BS_PER_GPU * N_GPUS * N_ACCUMULATION)) -ne ${TOTAL_BS} ]; then
    echo "TOTAL_BS should be divisible by N_GPUS * N_ACCUMULATION"
    exit 1
fi
echo "BS_PER_GPU: ${BS_PER_GPU}"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 accelerate launch \
    --config_file ../configs/ddp.yaml \
    train_eschernet_lora.py \
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
    --output_dir logs_node${node}_lora \
    --tracker_project_name eschernet

# ... 



















set +x; duration=SECONDS; RED='\033[0;31m'; Yellow='\033[1;33m'; Green='\033[0;32m'; NC='\033[0m'; echo -e "RED$((duration / 3600))hNC Yellow$((duration / 60 % 60))mNC Green$((duration % 60))sNC elapsed."