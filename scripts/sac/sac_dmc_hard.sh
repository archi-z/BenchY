#!/bin/sh
CUDA_VISIBLE_DEVICES=6
seed_max=4
agent="sac"
env="dmc_hard"
env_name="dog-run"
n_step=1

# Execute seed_max+1 random seeds (from 0 to seed_max, including seed_max)
for seed in `seq 0 ${seed_max}`;
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run.py \
        --overrides \
            seed=${seed} \
            agent=${agent} \
            env=${env} \
            env.env_name=${env_name} \
            n_step=${n_step}
done
