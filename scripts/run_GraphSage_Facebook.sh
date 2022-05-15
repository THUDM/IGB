#!/bin/bash

python main.py \
    --learning_rate_list 5e-3 --drop_rate_list 0.1 --weight_decay 1e-4 \
    --hidden_size_list 140 100 40 --num_layers_list 4 --sample_size_list 20 15 15 10 --aggr_list mean --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model GraphSage
