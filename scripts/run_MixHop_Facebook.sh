#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py \
    --learning_rate_list 0.007 --drop_rate_list 0.9 --weight_decay 0.0008 \
    --layer_pows_list 160 160 160 80 80 80 --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model MixHop
