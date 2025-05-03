#!/bin/sh
CUDA_VISIBLE_DEVICES=0
env_name="dog-run humanoid-run"
seed=4
n_step=3
encoder_horizon=5

for en in ${env_name};
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run_mr.py \
        --overrides \
            agent="mrddpg" \
            env="dmc_hard" \
            env.env_name=${en} \
            seed=${seed} \
            n_step=${n_step} \
            agent.encoder_horizon=${encoder_horizon} \
            updates_per_interaction_step=1
done