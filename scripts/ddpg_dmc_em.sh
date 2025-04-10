CUDA_VISIBLE_DEVICES=1 \
python run.py \
    --config_path ./configs \
    --config_name base \
    --overrides \
        seed=0 \
        agent=ddpg \
        env=dmc_em \
        env.env_name=quadruped-walk