#!/bin/bash

python main.py \
    --learning_rate_list 1e-4 --drop_rate_list 0.2 --hidden_size_list 128 --weight_decay 5e-5 \
    --num_layers_list 2 --num_iterations_list 8 --alpha_list 0.1 --lmbda_list 2e-3 --num_epochs 500 --eval_times 100 \
    --model PPNP \
    --seeds  \
    --dataset Flickr
done