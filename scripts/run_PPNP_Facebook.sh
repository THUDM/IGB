#!/bin/bash

python main.py \
    --learning_rate_list 0.03 --drop_rate_list 0.05 --hidden_size_list 512 --weight_decay 1e-4 \
    --num_layers_list 2 --num_iterations_list 8 --alpha_list 0.1 --lmbda_list 0.005 --eval_times 100 \
    --model PPNP \
    --seeds 1 \
    --dataset Facebook
