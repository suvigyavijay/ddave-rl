# from stable_baselines import  DQN,PPO
# from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
# from stable_baselines3.common.env_checker import check_env
from env_new import DangerousDaveEnv
import time, os
import argparse
# from custom_cnn import policy_kwargs
import torch
import numpy as np
# from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize,SubprocVecEnv,VecFrameStack
from stable_baselines3.common.env_checker import check_env

from rainbow import DQNAgent



if __name__ == "__main__":
    device = 'cuda'
    device = torch.device(device)

    # Manual assignment of arguments (replace with your desired values or use ipywidgets for interactivity)
    train = True  # equivalent to --train in argparse
    evaluate = False  # equivalent to --evaluate in argparse
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
    num_env = 16
    if num_env > 1:
        env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy) for _ in range(num_env)])
        # env = VecFrameStack(env,4)
    else:
        env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy)
        # stacked = FrameStack(env, num_stack=4,lz4_compress=False)


    eval_env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy) 
    total_timesteps=500000

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
            lr = 0.0001
            observation_space_dim = 209
            action_space_dim = 6
            
            if retrain:
                model = DQNAgent(env,num_env,eval_env=eval_env,observation_space_dim=observation_space_dim,action_space_dim=action_space_dim,
                memory_size=memory_size,batch_size=batch_size,target_update=target_update,
                             seed=42,n_step=n_step,device=device,v_min=v_min,v_max=v_max,
                             atom_size=atom_size,learning_rate=lr)

                model.load("checkpoints/{}".format(model_name))
            else:
                model = DQNAgent(env,num_env,eval_env=eval_env,observation_space_dim=observation_space_dim,action_space_dim=action_space_dim,
                memory_size=memory_size,batch_size=batch_size,target_update=target_update,
                             seed=42,n_step=n_step,device=device,v_min=v_min,v_max=v_max,
                             atom_size=atom_size,learning_rate=lr)
               

            model.learn(total_timesteps)
            env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,
                random_respawn=False,policy=policy) for _ in range(num_env)])
            model.env = env
            model.learn(total_timesteps)
            model.save("checkpoints/{}".format(model_name))
          
        if evaluate:
            model = DQNAgent(env,memory_size=memory_size,batch_size=batch_size,target_update=target_update,seed=42,
                             n_step=n_step,device=device,v_min=-2000,v_max=2000,atom_size=atom_size)
            # Evaluate the trained model
            model.load("checkpoints/{}".format(model_name))

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

      
            