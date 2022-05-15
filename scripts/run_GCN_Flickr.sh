#!/bin/bash

python main.py \
    --learning_rate_list 0.005 --drop_rate_list 0.4 --hidden_size_list 64 \
    --weight_decay 2e-4 --eval_times 100 \
    --dataset Flickr \
    --seeds 1 \
    --model GCN