import os
import argparse

import torch

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_envs
from utils import set_config, train_off_policy_mr

# Limit CPU usage
cpu_num = 4
os.environ['OMP_NUM_THREADS']=str(cpu_num)
os.environ['MKL_NUM_THREADS']=str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS']=str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS']=str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS']=str(cpu_num)
torch.set_num_threads(cpu_num)


def run(args):
    cfg = set_config(args)

    train_env, eval_env = create_envs(**cfg.env)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    buffer = create_buffer(
        observation_space=observation_space, action_space=action_space, **cfg.buffer
    )
    buffer.reset()

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    logger = WandbTrainerLogger(cfg)

    train_off_policy_mr(
        cfg=cfg,
        train_env=train_env,
        eval_env=eval_env,
        agent=agent,
        buffer=buffer,
        logger=logger
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--overrides", nargs="+", default=[])
    args = parser.parse_args()

    run(vars(args))
