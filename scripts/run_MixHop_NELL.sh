#!/bin/bash

python main.py \
    --learning_rate_list 0.05 --drop_rate_list 0.5 --weight_decay 2e-4 \
    --layer_pows_list 120 120 120 60 60 60 --eval_times 100 \
    --dataset NELL \
    --seeds 1 \
    --model MixHop
