#!/bin/bash

# Halfcheetah-medium-replay

uv run experiment.py --seed 123 \
    --env halfcheetah --dataset medium-replay --dataset_postfix filtered_10%   \
    --eta 0.4 --grad_norm 15.0 --model_type bc \
    --exp_name bc_stochastic --project_name qt_baseline --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 10000 --lr_decay \
    --early_stop --k_rewards --use_discount --K 5 -w true --batch_size 64 \
    --n_layer 3 --embed_dim 128 --learning_rate 1e-4 --stochastic_policy  \

uv run experiment.py --seed 123 \
    --env halfcheetah --dataset medium-replay   \
    --eta 5.0 --grad_norm 15.0 --model_type qdt \
    --exp_name qt --project_name qt_baseline --save_path ./save/  \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --K 5 -w true \

uv run experiment.py --seed 123 \ 
    --env halfcheetah --dataset medium-expert --use_aug --pct_traj 0.1  \
    --alpha 0.01 --eta2 1.0 --eta 0.4 --grad_norm 15.0 --model_type qdt \
    --exp_name erqt-gaussian-no-lr-decay --project_name erqt --save_path ./save/ \
    --max_iters 500 --num_steps_per_iter 10000 \
    --early_stop --k_rewards --use_discount --K 5 -w true \
    --policy_penalty --stochastic_policy --n_layer 3 --embed_dim 128 \
    --behavior_ckpt_file ./save/10%_bc_stochastic-halfcheetah-medium-replay-123-250324-112957/epoch_16.pth \