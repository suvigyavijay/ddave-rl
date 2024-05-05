import configparser
import time, os
import argparse
import pygame

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers.frame_stack import FrameStack
from env import DangerousDaveEnv
from algos.utils import RecordEpisodeStatistics

from algos.ppo import PPO
from algos.rnd import RND
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
    argparser.add_argument("--model-type", action="store", default="rnd", choices=["ppo", "rnd"], help="Choose the model type")
    argparser.add_argument("--model-load-path", action="store", help="Provide path to load model")
    args = argparser.parse_args()
    if args.model_type is None:
        print("Please specify the model type")
        exit(1)

    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "{}_ddave_{}".format(args.model_type, checkpoint_timestamp)
        
    # Create the DangerousDaveEnv environment
    envs = SubprocVecEnv([lambda : FrameStack(DangerousDaveEnv(env_rep_type="image"), 4) for _ in range(NUM_ENVS)])
    eval_env = DummyVecEnv([lambda : FrameStack(DangerousDaveEnv(env_rep_type="image"), 4)])

    envs.num_envs = NUM_ENVS
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    
    algo = None
    if args.model_type == "ppo":
        total_timesteps = int(config['PPO']['TOTAL_TIMESTEPS'])
        algo = PPO(envs, eval_env, model_name, total_timesteps=total_timesteps, num_envs=NUM_ENVS, num_steps=NUM_STEPS)
        
    elif args.model_type == "rnd":
        total_timesteps = int(config['RND']['TOTAL_TIMESTEPS'])
        algo = RND(envs, eval_env, model_name, total_timesteps=total_timesteps, num_envs=NUM_ENVS, num_steps=NUM_STEPS)

    if args.model_load_path is not None:
        algo.load_checkpoint(args.model_load_path)

    if args.train:
        algo.train()

    if args.evaluate:
        # load latest model
        algo.load_checkpoint()
        algo.evaluate("trained")
        
