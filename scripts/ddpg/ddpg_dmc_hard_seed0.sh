#!/bin/sh
CUDA_VISIBLE_DEVICES=6
env_name="dog-run humanoid-run"
seed=0
n_step=1

for en in ${env_name};
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run.py \
        --overrides \
            agent="ddpg" \
            env="dmc_hard" \
            env.env_name=${en} \
            seed=${seed} \
            n_step=${n_step}
done