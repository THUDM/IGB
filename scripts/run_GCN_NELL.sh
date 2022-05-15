#!/bin/bash

python main.py \
    --learning_rate_list 0.1 --drop_rate_list 0.4 --hidden_size_list 256 \
    --weight_decay 1e-4 --eval_times 100 \
    --dataset NELL \
    --seeds 1 \
    --model GCN