#!/bin/bash

python main.py \
--learning_rate_list 0.02 --drop_rate_list 0.2 --hidden_size_list 128 \
--weight_decay 1e-3 --nhead_list 16 --attn_drop_list 0.3 --num_epochs 150 --eval_times 100 \
--dataset AMiner \
--seeds 1 \
--model GAT
