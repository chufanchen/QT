#!/bin/bash

## Augment dataset
# Download d4rl dataset
python D4RL/create_dataset.py

python main.py run_params=aug env_params=halfcheetah_medium env_params.pct_traj=0.1

python main.py run_params=aug env_params=maze2d_umaze env_params.pct_traj=0.1

python main.py run_params=aug env_params=maze2d_medium env_params.pct_traj=0.1

python main.py run_params=aug env_params=maze2d_large env_params.pct_traj=0.1


## Filtered BC

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=halfcheetah_medium

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=halfcheetah_medium_expert

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=halfcheetah_medium_replay

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=hopper_medium

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=hopper_medium_expert

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=hopper_medium_replay

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium_expert

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium_replay

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium_expert_replay

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium_replay_expert

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=walker2d_medium_expert_replay_expert

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=maze2d_large

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=maze2d_umaze

python main.py agent_params=bc env_params.pct_traj=0.1 env_params=maze2d_medium

#########################

## ERQT

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=halfcheetah_medium

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=halfcheetah_medium_expert

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=halfcheetah_medium_replay

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=hopper_medium

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=hopper_medium_expert

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=hopper_medium_replay

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium_expert

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium_replay

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium_expert_replay

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium_replay_expert

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=walker2d_medium_expert_replay_expert

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=maze2d_large run_params.num_steps_per_iter=1000 run_params.eta=4.0 run_params.max_iters=100 run_params.early_epoch=50

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=maze2d_umaze run_params.num_steps_per_iter=1000 run_params.eta=5.0 run_params.max_iters=100 run_params.early_epoch=50

python main.py agent_params=erqt env_params.use_aug=true env_params.pct_traj=0.1 env_params=maze2d_medium run_params.num_steps_per_iter=1000 run_params.eta=5.0 run_params.max_iters=100 run_params.early_epoch=50

#########################

## ERQT hyperparameter sweep

python launch_sweep.py --config-name sweep/halfcheetah_medium_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/halfcheetah_medium_expert_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/halfcheetah_medium_replay_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/hopper_medium_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/hopper_medium_expert_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/hopper_medium_replay_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/walker2d_medium_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/walker2d_medium_expert_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

python launch_sweep.py --config-name sweep/walker2d_medium_replay_sweep.yaml env_params.use_aug=true env_params.pct_traj=0.1 -m

#########################


python experiment.py --seed 123 \
    --env pen --dataset human   \
    --eta 0.1 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \

python experiment.py --seed 123 \
    --env pen --dataset cloned   \
    --eta 0.1 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \

python experiment.py --seed 123 \
    --env hammer --dataset human   \
    --eta 0.1 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env hammer --dataset cloned   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 30 \

python experiment.py --seed 123 \
    --env door --dataset human   \
    --eta 0.005 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env door --dataset cloned   \
    --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env kitchen --dataset complete   \
    --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 100 \

python experiment.py --seed 123 \
    --env kitchen --dataset partial   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open-dense   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset umaze   \
    --eta 5.0 --grad_norm 20.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset umaze-dense   \
    --eta 3.0 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium   \
    --eta 5.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium-dense   \
    --eta 5.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large   \
    --eta 4.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large-dense   \
    --eta 4.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze   \
    --eta 0.05 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze-diverse   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset medium-diverse   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --early_stop --k_rewards --use_discount  --early_epoch 80 \

python experiment.py --seed 123 \
    --env antmaze --dataset large-diverse   \
    --eta 0.005 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --early_stop --k_rewards --use_discount  --early_epoch 80 --reward_tune cql_antmaze 
