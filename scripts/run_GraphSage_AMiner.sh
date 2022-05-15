#!/bin/bash

python main.py \
    --learning_rate_list 0.02 --drop_rate_list 0 --weight_decay 5e-5 \
    --hidden_size_list 512 --num_layers_list 2 --sample_size_list 10 10 --aggr_list mean --eval_times 100 \
    --dataset AMiner \
    --seeds 1 \
    --model GraphSage
