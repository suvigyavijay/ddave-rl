from stable_baselines3 import  DQN,PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize,SubprocVecEnv,VecFrameStack
from stable_baselines3.common.env_checker import check_env
from env_new import DangerousDaveEnv
import time, os
import argparse
from custom_cnn import cnn_policy_kwargs,mlp_policy_kwargs
import torch
import numpy as np
from stable_baselines3.ppo import MlpPolicy as MLP_PPO
from stable_baselines3.dqn import MlpPolicy as MLP_DQN
import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback,CallbackList
from qlearning import NStepDoubleQLearningAgent
import pickle
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.vec_env import VecFrameStack
from rllte.xplore.reward import RND,PseudoCounts
from rllte_core import RLeXploreCallback 

if __name__ == "__main__":

 
    device = 'cuda'
    device = torch.device(device)

    # Manual assignment of arguments (replace with your desired values or use ipywidgets for interactivity)
    train = True  # equivalent to --train in argparse
    evaluate = False  # equivalent to --evaluate in argparse
    model_name = "DQN"  # manually specify or generate a name
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
    random_respawn=True
    policy = 'MLP'
    num_env = 16
    if num_env > 1:
        env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy) for _ in range(num_env)])
        # env = VecFrameStack(env,4)
    else:
        env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=random_respawn,policy=policy)
        # stacked = FrameStack(env, num_stack=4,lz4_compress=False)
        
    total_timesteps=10000000

    eval_env = DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy)
    # eval_env = SubprocVecEnv([lambda : eval_env])
    # eval_env = VecFrameStack(eval_env,4)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10000,
                             deterministic=True, render=False)
  
    if model_type == 'DQN':
        if train:
            # Define and train the DQN agent
            if retrain:
                model = DQN.load("checkpoints/{}".format(model_name))
                model.set_env(env)
            else:
                
                model = DQN(MLP_DQN, env, verbose=1, batch_size=1024, policy_kwargs=mlp_policy_kwargs,
                            learning_starts=10000, exploration_fraction=1, exploration_final_eps=0.001, device=device,
                            target_update_interval=10000, buffer_size=5000000)


            model.learn(total_timesteps=total_timesteps, progress_bar=True,log_interval=1,callback=eval_callback)
            model.save("checkpoints/{}".format(model_name))

            env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", 
             env_rep_type=env_rep_type,random_respawn=False,policy=policy) for _ in range(num_env)])
           
            irs = PseudoCounts(env, device=device)
            explore_callback = RLeXploreCallback(irs)
            model.set_env(env)
            model.learn(total_timesteps=total_timesteps, progress_bar=True,log_interval=1,callback=eval_callback)
            # Save the trained model if desired
            model.save("checkpoints/{}".format(model_name))

        if evaluate:
            pass
            # Evaluate the trained model
            # model = DQN.load("checkpoints/{}".format(model_name), env=env)

    elif model_type == 'PPO':
        if train:
            # Define and train the PPO agent
            if retrain:
                model = PPO.load("checkpoints/{}".format(model_name), env=env)
            else:
                irs = RND(env, device=device)
                explore_callback = RLeXploreCallback(irs)
                model = PPO(MLP_PPO, env, verbose=1, batch_size=512, device=device,
                        n_steps=2048,policy_kwargs=mlp_policy_kwargs,ent_coef=0.5)
        
            callback = CallbackList([explore_callback, eval_callback])
            model.learn(total_timesteps=total_timesteps, progress_bar=True,log_interval=1,callback=callback)
            model.save("checkpoints/{}".format(model_name))
            
        if evaluate:
            # Evaluate the trained model
            model = PPO.load("checkpoints/{}".format(model_name), env=env)
    
    elif model_type == 'DQ':
        if train:
            learning_rate = 0.0001
            max_epsilon = 1
            min_epsilon = 0.0001
            epsilon_decay_rate = (min_epsilon/max_epsilon)**(1/total_timesteps)
            discount_factor = 0.99
            n_steps = 100
            observation_space_dim = eval_env.observation_space.shape[0]
            action_space_dim = 4
            if retrain:
                model = NStepDoubleQLearningAgent(num_env=num_env,env=env,eval_env=eval_env,action_space_dim=action_space_dim,observation_space_dim=observation_space_dim,
                max_epsilon=max_epsilon,min_epsilon=min_epsilon,epsilon_decay_rate=epsilon_decay_rate,discount_factor=discount_factor,
                       learning_rate=learning_rate,n_steps=n_steps)
                model.load_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')
                model.train(steps=total_timesteps,verbose=True)


                model.save_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')
                
                env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy) for _ in range(num_env)])

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
                
                env = SubprocVecEnv([lambda : DangerousDaveEnv(render_mode="human", env_rep_type=env_rep_type,random_respawn=False,policy=policy) for _ in range(num_env)])

                model = NStepDoubleQLearningAgent(num_env=num_env,env=env,eval_env=eval_env,action_space_dim=action_space_dim,observation_space_dim=observation_space_dim,
                max_epsilon=max_epsilon,min_epsilon=min_epsilon,epsilon_decay_rate=epsilon_decay_rate,discount_factor=discount_factor,
                       learning_rate=learning_rate,n_steps=n_steps)
                model.load_model('q_table_a_ddave.pkl','q_table_b_ddave.pkl')
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