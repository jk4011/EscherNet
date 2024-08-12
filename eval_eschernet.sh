#!/bin/bash
# bash script to evaluate the model

# TODO ################### Chose CaPE type  ##########################

GPU=$1

# 6DoF
cape_type="6DoF"
pretrained_model="/data2/wlsgur4011/eschernet_data/ckpt/eschernet-6dof"

## 4DoF
#cape_type="4DoF"
#pretrained_model="/data2/wlsgur4011/eschernet_data/ckpt/eschernet-4dof"

################### Chose CaPE type  ##########################

# TODO ###################  Chose data type  ##########################

# demo
# data_type="GSO25"
# T_ins=(1 2 3 5 10)
# data_dir="./demo/GSO30"

## GSO
# data_type="GSO25" # GSO25, GSO3D, GSO100, NeRF, RTMV
# T_ins=(1 2 3 5 10)
# data_dir="/data2/wlsgur4011/eschernet_data/EscherNet-Dataset/GSO30"

## RTMV
data_type="RTMV"
T_ins=(1 2 3 5 10)
data_dir="/data2/wlsgur4011/eschernet_data/RTMV/"

## NeRF
#data_type="NeRF"
#T_ins=(1 2 3 5 10 20 50 100)
#data_dir="/data2/wlsgur4011/eschernet_data/nerf_synthetic/nerf_synthetic"

## Real World Franka Recordings
#data_type="Franka"
#T_ins=(5)
#data_dir="/data2/wlsgur4011/eschernet_data/EscherNet-Dataset/Franka16/"

## MVDream, 4 views to 100
#data_type='MVDream'
#T_ins=(4)
#data_dir="/data2/wlsgur4011/eschernet_data/EscherNet-Dataset/MVDream"

## Text2Img, 1 view to 100
#data_type='Text2Img'
#T_ins=(1)
#data_dir="/data2/wlsgur4011/eschernet_data/EscherNet-Dataset/Text2Img/"

###################  Chose data type  ##########################


# run
for T_in in "${T_ins[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python eval_eschernet.py --pretrained_model_name_or_path "$pretrained_model" \
                         --data_dir "$data_dir" \
                         --data_type "$data_type" \
                          --cape_type "$cape_type" \
                         --T_in "$T_in"
done
