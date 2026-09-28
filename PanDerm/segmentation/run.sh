#!/bin/bash

echo "Running PanDerm experiment on ISIC18"
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --worker 8 --gpu "0,1" --batch_size 1 --test_batch_size 1 \
    --epoch 100 --lr 1e-4 --weight_decay 0.05 \
    --log_name "cae_seg_lr_1e-4_decay_0.05_full" --model "cae_seg" --size 224 \
    --dataset "ISIC2018" \
    --parent_path "/mount/neuron/Lamborghini/dir/pythonProject/agent_project/" \
    --save_name "out" \
    --seed 0 --smoke


#
#    --dataset_path "/mount/neuron/Lamborghini/dir/pythonProject/agent_project/ISIC2018/" \

