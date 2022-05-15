#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0 --hidden_size_list 128 --weight_decay 2e-4 \
    --num_layers_list 2 --num_iterations_list 6 --alpha_list 0.8 --lmbda_list 0.008 --eval_times 100 \
    --model PPNP \
    --seeds 1 \
    --dataset AMiner
