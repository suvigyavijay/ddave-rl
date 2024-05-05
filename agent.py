import configparser
import time, os
import argparse
import pygame

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers.frame_stack import FrameStack
from env import DangerousDaveEnv
from algos.utils import RecordEpisodeStatistics

from algos.ppo import PPO
config = configparser.ConfigParser()
config.read('algo.cfg')

SEED = int(config['COMMON']['SEED'])
NUM_ENVS = int(config['COMMON']['NUM_ENVS'])
NUM_STEPS = int(config['COMMON']['NUM_STEPS'])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the model")
    argparser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    argparser.add_argument("--model-name", action="store", help="Load the latest model")
    argparser.add_argument("--env-rep-type", action="store", default="image", choices=["text", "image", "grid"], help="Choose the environment representation type")
    argparser.add_argument("--model-type", action="store", default="dqn", choices=["dqn", "ppo", "rnd"], help="Choose the model type")
    args = argparser.parse_args()

    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "{}_ddave_{}".format(args.model_type, checkpoint_timestamp)
        
    # Create the DangerousDaveEnv environment
    envs = SubprocVecEnv([lambda : FrameStack(DangerousDaveEnv(env_rep_type="image"), 4) for _ in range(NUM_ENVS)])

    envs.num_envs = NUM_ENVS
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    
    algo = None
    if args.model_type == "ppo":
        total_timesteps = int(config['PPO']['TOTAL_TIMESTEPS'])
        algo = PPO(envs, model_name, total_timesteps=total_timesteps, num_envs=NUM_ENVS, num_steps=NUM_STEPS)

    if args.train:
        algo.train()

    if args.evaluate:
        # load latest model
        algo.load_checkpoint()
        algo.evaluate("trained")
        
