#!/bin/sh
CUDA_VISIBLE_DEVICES=6
env_name="humanoid-run"
seed=4
n_step=1
encoder_horizon=5


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python run_mr.py \
    --overrides \
        agent="mrddpg" \
        env="dmc_hard" \
        env.env_name=${env_name} \
        seed=${seed} \
        n_step=${n_step} \
        agent.encoder_horizon=${encoder_horizon}