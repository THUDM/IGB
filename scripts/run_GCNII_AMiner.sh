#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0 --hidden_size_list 128 --num_layers_list 16 \
    --alpha_list 0.9 --lmbda_list 0.7 --wd1 1e-3 --wd2 5e-4 --eval_times 100 \
    --dataset AMiner \
    --seeds 1 \
    --model GCNII