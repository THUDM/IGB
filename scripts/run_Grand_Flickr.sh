#!/bin/bash

python main.py \
    --learning_rate_list 0.01 --drop_rate_list 0.9 --hidden_size_list 32 \
    --weight_decay 5e-5 --order_list 6 --use_bn_list 1 --input_drop_list 0.5 \
    --dropnode_list 0 --lmbda_list 0.3 --temp_list 0.05 --sample_num_list 2 --eval_times 100 \
    --dataset Flickr \
    --seeds 1 \
    --model Grand