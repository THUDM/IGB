#!/bin/bash

python main.py \
    --learning_rate_list 0.05 --drop_rate_list 0.1 --hidden_size_list 64 \
    --weight_decay 2e-4 --nhead_list 12 --attn_drop_list 0.1 --num_epochs 600 --eval_times 100 \
    --dataset NELL \
    --seeds 1 \
    --model GAT
