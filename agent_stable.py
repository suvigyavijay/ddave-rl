# from stable_baselines import  DQN,PPO
# from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
# from stable_baselines3.common.env_checker import check_env
from env import DangerousDaveEnv
import time, os
import argparse
# from custom_cnn import policy_kwargs
import torch
import numpy as np
# from stable_baselines3.ppo import MlpPolicy,CnnPolicy

from rainbow import DQNAgent



if __name__ == "__main__":
    device = 'mps'
    device = torch.device(device)

    # Manual assignment of arguments (replace with your desired values or use ipywidgets for interactivity)
    train = True  # equivalent to --train in argparse
    evaluate = True  # equivalent to --evaluate in argparse
    model_name = "Rainbow_DQN"  # manually specify or generate a name
   
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
    random_respawn=True
    policy = 'MLP'
    env_rep_type = 'text'  # 'text' or 'image'
    env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy)
    obs,info = env.reset()
    total_timesteps=50000
    if model_type == 'DQN':
        if train:
            # Define and train the DQN agent
            memory_size = 100000
            batch_size = 512
            target_update = 100
            n_step = 10
            atom_size=1000
            v_min = -2000
            v_max = 2000

            if retrain:
                model = DQNAgent(env,memory_size=memory_size,batch_size=batch_size,target_update=target_update,
                             seed=42,n_step=n_step,device=device,v_min=-2000,v_max=2000,atom_size=atom_size)

                model.load("checkpoints/{}".format(model_name))
            else:
                model = DQNAgent(env,memory_size=memory_size,batch_size=batch_size,target_update=target_update,
                             seed=42,n_step=n_step,device=device,v_min=-2000,v_max=2000,atom_size=atom_size)

            model.learn(total_timesteps)
            env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy)
            model.env = env
            obs,info = env.reset()
            model.learn(total_timesteps)
            model.save("checkpoints/{}".format(model_name))
          
        if evaluate:
            model = DQNAgent(env,memory_size=memory_size,batch_size=batch_size,target_update=target_update,seed=42,
                             n_step=n_step,device=device,v_min=-2000,v_max=2000,atom_size=atom_size)
            # Evaluate the trained model
            model.load("checkpoints/{}".format(model_name))

    # elif model_type == 'PPO':
    #     if train:
    #         # Define and train the PPO agent
    #         if retrain:
    #             model = PPO.load("checkpoints/{}".format(model_name), env=env,tensorboard_log=tensorboard_log)
    #         else:
    #             model = PPO(MlpPolicy, env, verbose=1, batch_size=256, policy_kwargs=policy_kwargs, device=device,
    #                         tensorboard_log=tensorboard_log,ent_coef=0.001,n_steps=2048,gae_lambda=0.95)

    #         model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
    #         model.save("checkpoints/{}".format(model_name))
    #         # env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False)
    #         # model = PPO.load("checkpoints/{}".format(model_name),tensorboard_log=tensorboard_log)
    #         # model.set_env(env)
    #         # obs,info = env.reset()
    #         # model.learn(total_timesteps=total_timesteps, progress_bar=True,tb_log_name=tensorboard_log_run_name,log_interval=1)
    #         # # Save the trained model if desired
    #         # model.save("checkpoints/{}".format(model_name))
            
    #     if evaluate:
    #         # Evaluate the trained model
    #         model = PPO.load("checkpoints/{}".format(model_name), env=env,tensorboard_log=tensorboard_log)

    if evaluate:
        eps_reward = []
        for i in range(5):
            env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy)
            obs, info = env.reset()

            for j in range(10):
                obs, __,_,_,_ = env.step(env.action_space.sample())
            terminated = False
            truncated = False
            reward = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)
                print(action,rewards)
                reward += rewards
            eps_reward.append(reward)
        print(f'{np.mean(eps_reward)} eval reward mean')
        print(eps_reward)

        env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy)
        obs, info = env.reset()

        for j in range(10):
            obs, __,_,_,_ = env.step(env.action_space.sample())
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()
            