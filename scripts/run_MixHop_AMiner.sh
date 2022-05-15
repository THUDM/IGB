#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.1 --weight_decay 0.002 \
    --layer_pows_list 200 200 200 100 100 100 --eval_times 100 \
    --dataset AMiner \
    --seeds 1 \
    --model MixHop
