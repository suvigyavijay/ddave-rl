from stable_baselines3 import  DQN,PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from env_grid import DangerousDaveEnv
import time, os
import argparse
import torch
import numpy as np
from stable_baselines3.ppo import MlpPolicy as MLP_PPO
from stable_baselines3.dqn import MlpPolicy as MLP_DQN
import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from qlearning import NStepDoubleQLearningAgent
import pickle


if __name__ == "__main__":
    device = 'cuda'
    device = torch.device(device)

    # Manual assignment of arguments (replace with your desired values or use ipywidgets for interactivity)
    train = True  # equivalent to --train in argparse
    evaluate = False  # equivalent to --evaluate in argparse
    model_name = "DQ_Learning"  # manually specify or generate a name
    env_rep_type = 'text'  # 'text' or 'image'
    model_type = 'DQ'  # 'DQN', 'RND', or 'PPO'
    retrain = False  # equivalent to --retrain in argparse

    # Your existing logic below
    checkpoint_timestamp = int(time.time())
    if not model_name:
        model_name = "checkpoints/ddave_{}".format(checkpoint_timestamp)

    tensorboard_log = f"tensorboard_log/{model_name}"
    tensorboard_log_run_name = '0'
    print(model_name,tensorboard_log)
    # Create the DangerousDaveEnv environment
    random_respawn=True
    policy = 'MLP'
    num_env = 32
    if num_env > 1:
        env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type) for _ in range(num_env)])
    else:
        env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type)
    
    total_timesteps=300_000

    eval_env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False)
  
    if model_type == 'DQ':
        if train:
            learning_rate = 0.0001
            max_epsilon = 1
            min_epsilon = 0.001
            epsilon_decay_rate = (min_epsilon/max_epsilon)**(1/(total_timesteps*1.3))
            discount_factor = 0.99
            n_steps = 50
            observation_space_dim = eval_env.observation_space.shape[0]
            action_space_dim = eval_env.action_space.n
            if retrain:
                model = NStepDoubleQLearningAgent(num_env=num_env,env=env,eval_env=eval_env,action_space_dim=action_space_dim,observation_space_dim=observation_space_dim,
                max_epsilon=max_epsilon,min_epsilon=min_epsilon,epsilon_decay_rate=epsilon_decay_rate,discount_factor=discount_factor,
                       learning_rate=learning_rate,n_steps=n_steps)
                model.load_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')
                model.train(steps=total_timesteps,verbose=True)

                model.save_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')

            else:
                model = NStepDoubleQLearningAgent(num_env=num_env,env=env,eval_env=eval_env,action_space_dim=action_space_dim,observation_space_dim=observation_space_dim,
                max_epsilon=max_epsilon,min_epsilon=min_epsilon,epsilon_decay_rate=epsilon_decay_rate,discount_factor=discount_factor,
                       learning_rate=learning_rate,n_steps=n_steps)
                model.train(steps=total_timesteps,verbose=True)
                model.save_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')
                


    if evaluate:
        eps_reward = []
        for i in range(5):
            env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy)
            obs, info = env.reset()
            terminated = False
            truncated = False
            reward = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = env.step(action)
                reward += rewards
            eps_reward.append(reward)
        print(f'{np.mean(eps_reward)} eval reward mean')
        print(eps_reward)