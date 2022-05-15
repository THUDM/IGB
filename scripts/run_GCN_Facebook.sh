#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.1 --hidden_size_list 64 \
    --weight_decay 4e-3 --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model GCN