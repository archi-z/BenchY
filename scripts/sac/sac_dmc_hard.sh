#!/bin/sh
CUDA_VISIBLE_DEVICES=4
config_path="./configs"
config_name="base"
seed_max=0
agent="sac"
env="dmc_hard"
env_name="dog-run"

# Execute seed_max+1 random seeds (from 0 to seed_max, including seed_max)
for seed in `seq 0 ${seed_max}`;
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run.py \
        --config_path ${config_path} \
        --config_name ${config_name} \
        --overrides \
            seed=${seed} \
            agent=${agent} \
            env=${env} \
            env.env_name=${env_name}
done
