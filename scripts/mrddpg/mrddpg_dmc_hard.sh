#!/bin/sh
CUDA_VISIBLE_DEVICES=7
seed_max=4
agent="mrddpg"
env="dmc_hard"
env_name="dog-run"
n_step=1
encoder_horizon=5

# Execute seed_max+1 random seeds (from 0 to seed_max, including seed_max)
for seed in `seq 0 ${seed_max}`;
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run_mr.py \
        --overrides \
            seed=${seed} \
            agent=${agent} \
            env=${env} \
            env.env_name=${env_name} \
            n_step=${n_step} \
            agent.encoder_horizon=${encoder_horizon}
done
