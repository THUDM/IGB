#!/bin/bash

python main.py \
    --learning_rate_list 8e-4 --drop_rate_list 0.3 --hidden_size_list 64 --num_layers_list 64 \
    --alpha_list 0.2 --lmbda_list 0.1 --wd1 1e-3 --wd2 5e-4 --eval_times 100 \
    --dataset Flickr \
    --seeds 1 \
    --model GCNII