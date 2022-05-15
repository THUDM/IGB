#!/bin/bash

python main.py \
    --learning_rate_list 0.02 --drop_rate_list 0.4 --hidden_size_list 512 \
    --weight_decay 5e-4 --order_list 1 --use_bn_list 0 --input_drop_list 0.1 \
    --dropnode_list 0.1 --lmbda_list 0.6 --temp_list 0.5 --sample_num_list 2 --eval_times 100 \
    --dataset AMiner \
    --seeds 1 \
    --model Grand