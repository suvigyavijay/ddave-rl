import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import Optional,Tuple
import copy
import pickle
import pandas as pd
import os
# Imports
import gymnasium
import math
import matplotlib.pyplot as plt


import numpy as np
import gymnasium as gym
import pickle

class NStepDoubleQLearningAgent:
    def __init__(self, num_env,env, eval_env,observation_space_dim,action_space_dim,max_epsilon: float, min_epsilon: float,
                 epsilon_decay_rate: float, discount_factor: float,
                 learning_rate: float, n_steps: int):
        self.num_env = num_env
        self.env = env
        self.eval_env = eval_env
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.epsilon = max_epsilon
        
        self.q_table_a = np.zeros((self.observation_space_dim, action_space_dim))
        self.q_table_b = np.zeros((self.observation_space_dim, action_space_dim))
        
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        
        self.reward_per_episode = []
        self.timesteps_per_episode = []

    def get_agent_position(self, obs):
        return np.argwhere(obs == 7)[:, 1] 
    
    def step(self, obs,explore=True):
        
        if explore:
            agent_positions = self.get_agent_position(obs)

            # Random choice for each environment to decide if it should explore or exploit
            explore = np.random.rand(self.num_env) <= self.epsilon

            actions = np.where(explore, np.random.randint(0,self.action_space_dim,size=(self.num_env,)),0)  # Random actions or zero

            # Calculate the best action for non-exploring instances
            # We stack Q-values for efficient computation
            combined_q_values = self.q_table_a[agent_positions, :] + self.q_table_b[agent_positions, :]
            best_actions = np.argmax(combined_q_values, axis=1)
            actions = np.where(explore, actions, best_actions)  # Replace zeros with best actions for non-explorers
        else:
            
            agent_positions = np.argwhere(obs == 7)[0]
            combined_q_values = self.q_table_a[agent_positions, :] + self.q_table_b[agent_positions, :]
            actions = np.argmax(combined_q_values, axis=1)

        return actions, agent_positions


    
    def add_to_buffer(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        if len(self.state_buffer) > self.n_steps:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
    
    def update_q_table_n_step(self):
        # Calculate the cumulative discounted rewards for each environment
        discounts = self.discount_factor ** np.arange(len(self.reward_buffer))
        G = np.sum([self.reward_buffer[i] * discounts[i] for i in range(len(self.reward_buffer))], axis=0)
        
        current_states = self.state_buffer[0]
        current_actions = self.action_buffer[0]
        next_states = self.state_buffer[-1]
        
        # Randomly choose which Q-table to update
        update_a = np.random.rand(len(next_states)) < 0.5
        
        # Compute the max actions from the appropriate tables
        max_actions_from_a = np.argmax(self.q_table_a[next_states], axis=1)
        max_actions_from_b = np.argmax(self.q_table_b[next_states], axis=1)

        # Update Q-table A for some environments and Q-table B for others
        for i in range(len(next_states)):
            if update_a[i]:
                self.q_table_a[current_states[i], current_actions[i]] += self.learning_rate * (
                    G[i] + (self.discount_factor ** self.n_steps) * self.q_table_b[next_states[i], max_actions_from_a[i]] - self.q_table_a[current_states[i], current_actions[i]])
            else:
                self.q_table_b[current_states[i], current_actions[i]] += self.learning_rate * (
                    G[i] + (self.discount_factor ** self.n_steps) * self.q_table_a[next_states[i], max_actions_from_b[i]] - self.q_table_b[current_states[i], current_actions[i]])

    def train(self, steps= 10000,verbose=False):

        obs  = self.env.reset()

        for step in range(1, steps + 1):

            actions, agent_pos = self.step(obs)
            next_obs, rewards, dones,info = self.env.step(actions)
            self.add_to_buffer(agent_pos, actions, rewards)
            
            if len(self.reward_buffer) == self.n_steps:
                self.update_q_table_n_step()
            
            obs = next_obs

            self.epsilon = max(self.min_epsilon, self.max_epsilon*(self.epsilon_decay_rate**step))
            if verbose and step % 20000==0:
                print(f'Steps: {step}, Epsilon: {self.epsilon}')
                print(self.evaluate())

    def save_model(self, path_a, path_b):
        with open(path_a, 'wb') as f_a:
            pickle.dump(self.q_table_a, f_a)
        with open(path_b, 'wb') as f_b:
            pickle.dump(self.q_table_b, f_b)
    

    def load_model(self, path_a, path_b):
        with open(path_a, 'rb') as f_a:
            self.q_table_a= pickle.load(f_a)
        with open(path_b, 'rb') as f_b:
            self.q_table_b= pickle.load(f_b)
    
    def evaluate(self,num_episodes=1):
        print('Evaluating Result')
        total_rewards = []
        for _ in range(num_episodes):
            obs,info = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            while not (terminated or truncated):
                action, _ = self.step(obs, explore=False)  # Modify the step function to include an 'explore' parameter
                obs, reward, terminated,truncated, _ = self.eval_env.step(action[0])
                episode_reward += reward
            total_rewards.append(episode_reward)
        average_reward = np.mean(total_rewards)
        print('Evaluating Result',total_rewards)
        return average_reward
