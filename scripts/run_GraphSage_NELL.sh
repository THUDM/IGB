#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.7 --weight_decay 5e-5 \
    --hidden_size_list 256 256 --num_layers_list 3 --sample_size_list 20 15 10 --aggr_list mean --eval_times 100 \
    --dataset NELL \
    --seeds 1 \
    --model GraphSage
