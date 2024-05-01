from stable_baselines3 import  DQN,PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from env_new import DangerousDaveEnv
import time, os
import argparse
from custom_cnn import policy_kwargs
import torch
import numpy as np
from stable_baselines3.ppo import MlpPolicy as MLP_PPO
from stable_baselines3.dqn import MlpPolicy as MLP_DQN
import random
from stable_baselines3.common.env_checker import check_env



if __name__ == "__main__":

    device = 'cuda'
    device = torch.device(device)

    # Manual assignment of arguments (replace with your desired values or use ipywidgets for interactivity)
    train = True  # equivalent to --train in argparse
    evaluate = True  # equivalent to --evaluate in argparse
    model_name = "dqn_test_2"  # manually specify or generate a name
    env_rep_type = 'text'  # 'text' or 'image'
    model_type = 'DQN'  # 'DQN', 'RND', or 'PPO'
    retrain = False  # equivalent to --retrain in argparse

    # Your existing logic below
    checkpoint_timestamp = int(time.time())
    if not model_name:
        model_name = "checkpoints/ddave_{}".format(checkpoint_timestamp)

    tensorboard_log = f"tensorboard_log/{model_name}"
    tensorboard_log_run_name = '0'
    print(model_name,tensorboard_log)
    # Create the DangerousDaveEnv environment
    random_respawn=False
    policy = 'MLP'
    num_env = 16
    if num_env > 1:
        env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy) for _ in range(num_env)])
    else:
        env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy)
    
    total_timesteps=5000000
  
    if model_type == 'DQN':
        if train:
            # Define and train the DQN agent
            if retrain:
                model = DQN.load("checkpoints/{}".format(model_name),tensorboard_log=tensorboard_log)
                model.set_env(env)
            else:
                model = DQN(MLP_DQN, env, verbose=1, batch_size=512, policy_kwargs=policy_kwargs,
                            learning_starts=10000, exploration_fraction=0.6, exploration_final_eps=0.01, device=device,
                            target_update_interval=10000, buffer_size=100000,tensorboard_log=tensorboard_log)

            model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
            model.save("checkpoints/{}".format(model_name))

            # env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False)
            # model = DQN.load("checkpoints/{}".format(model_name),tensorboard_log=tensorboard_log)
            # model.set_env(env)
            # env.reset()
            # model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
            # # Save the trained model if desired
            # model.save("checkpoints/{}".format(model_name))

        if evaluate:
            # Evaluate the trained model
            model = DQN.load("checkpoints/{}".format(model_name), env=env,tensorboard_log=tensorboard_log)

    elif model_type == 'PPO':
        if train:
            # Define and train the PPO agent
            if retrain:
                model = PPO.load("checkpoints/{}".format(model_name), env=env,tensorboard_log=tensorboard_log)
            else:
                model = PPO(MLP_PPO, env, verbose=1, batch_size=256, policy_kwargs=policy_kwargs, device=device,
                            tensorboard_log=tensorboard_log,ent_coef=0.001,n_steps=2048,gae_lambda=0.95)

            model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
            model.save("checkpoints/{}".format(model_name))
            # env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False)
            # model = PPO.load("checkpoints/{}".format(model_name),tensorboard_log=tensorboard_log)
            # model.set_env(env)
            # obs,info = env.reset()
            # model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
            # # Save the trained model if desired
            # model.save("checkpoints/{}".format(model_name))
            
        if evaluate:
            # Evaluate the trained model
            model = PPO.load("checkpoints/{}".format(model_name), env=env,tensorboard_log=tensorboard_log)

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