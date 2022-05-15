#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.7 --hidden_size_list 128 \
    --weight_decay 6e-5 --order_list 20 --use_bn_list 1 --input_drop_list 0.4 \
    --dropnode_list 0 --lmbda_list 1 --temp_list 0.7 --sample_num_list 10 --eval_times 100 \
    --dataset Facebook \
    --seeds 1 \
    --model Grand