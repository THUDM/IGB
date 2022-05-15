#!/bin/bash

python main.py \
    --learning_rate_list 0.03 --drop_rate_list 0 --hidden_size_list 64 \
    --weight_decay 5e-4 --eval_times 100 \
    --dataset AMiner \
    --seeds 1 \
    --model GCN \
