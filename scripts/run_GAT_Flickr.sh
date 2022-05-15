#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.3 --hidden_size_list 32 --GAT_normal 1 \
    --weight_decay 2e-4 --nhead_list 16 --attn_drop_list 0.4 --num_epochs 500 --eval_times 100 \
    --dataset Flickr \
    --seeds 1 \
    --model GAT
