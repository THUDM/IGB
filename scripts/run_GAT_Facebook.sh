#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.5 --hidden_size_list 64 \
    --weight_decay 1e-4 --nhead_list 8 --attn_drop_list 0.3 --num_epochs 800 --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model GAT
