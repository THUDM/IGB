#!/bin/bash

python main.py \
    --learning_rate_list 5e-4 --drop_rate_list 0 --weight_decay 0.0004 \
    --layer_pows_list 110 110 110 40 40 40 --best_epoch 1 --eval_times 100 \
    --model MixHop \
    --seeds 1 \
    --dataset Flickr
