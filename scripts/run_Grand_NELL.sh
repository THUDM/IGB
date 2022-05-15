#!/bin/bash

python main.py \
    --learning_rate_list 0.01  --drop_rate_list 0.3 --hidden_size_list 32 \
    --weight_decay 5e-4 --order_list 8 --use_bn_list 0 --input_drop_list 0.6 \
    --dropnode_list 0.5 --lmbda_list 1 --temp_list 0.5 --sample_num_list 2 --num_epochs 50 --eval_times 100 \
    --dataset NELL \
    --seeds 1 \
    --model Grand
