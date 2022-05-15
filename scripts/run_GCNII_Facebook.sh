#!/bin/bash

python main.py \
    --learning_rate_list 0.02 --drop_rate_list 0.2 --hidden_size_list 64 --num_layers_list 256 \
    --alpha_list 0.2 --lmbda_list 1e-5 --wd1 1e-3 --wd2 5e-4 --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model GCNII