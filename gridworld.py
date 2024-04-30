# Defining Base GridEnviroment Class for Deterministic and Stochatic Enviroment
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import Optional,Tuple
import copy
import pickle
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle
class BaseGridEnvironment(gym.Env):

    # Attribute of a Gym class that provides info about the render modes
    metadata = { 'render.modes': [] }

    # Initialization function
    def __init__(self,grid_size :Tuple[int,int],max_timesteps: int):
        
        #intializing obseration space and action space
        self.grid_size = grid_size
        self.observation_space = spaces.Discrete(self.grid_size[0]*self.grid_size[1])
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = max_timesteps

        _,_ = self.reset(None)
      

    def are_same_pos(self,pos1,pos2):
        return pos1[0] == pos2[0] and pos1[1]==pos2[1]
    
    def action_to_string(self,action):
        if action == 0:
            return 'Down'
        elif action == 1:
            return  'UP'
        elif action == 2:
            return 'RIGHT'
        
        elif action == 3:
            return 'LEFT'
        else:
            return 'NONE'
        
    def create_update_grid(self):

        self.state = np.zeros(self.grid_size)
    
        self.state[tuple(self.goal_pos)] = 5
        self.state[tuple(self.quick_sand_pos)] = 15
        if self.diamond_pos:
            self.state[tuple(self.diamond_pos)] = 35
        self.state[tuple(self.agent_pos)] = 25
        
        # if self.last_agent_pos is not None:
        #     if not self.are_same_pos(self.last_agent_pos,self.agent_pos):
        #         self.state[tuple(self.last_agent_pos)] = 45
        
        

    # Reset function and place agent at random place which is not equal to goal_pos
    def reset(self, seed: Optional[int]=None):
        
        
        self.goal_pos  = [self.grid_size[0]-1,self.grid_size[1]-1]
        self.quick_sand_pos = [self.grid_size[0]-2,self.grid_size[1]-2]
        self.diamond_pos = [1,self.grid_size[1]-2]
        self.diamond_pos_copy =  [1,self.grid_size[1]-2]
        self.last_agent_pos = None
        self.time_steps = 0
        self.last_action = 'None'
        self.last_reward = 'None'
        
    
        if seed is None:
            self.agent_pos = [0,0]
        else:
            np.random.seed(seed)
            # doesnt allow the agent to spawn at the same postion as trophy,quicksand and diamond
            while True:
                self.agent_pos = [np.random.randint(0,self.grid_size[0]),np.random.randint(0,self.grid_size[1])]
                if not self.are_same_pos(self.agent_pos,self.goal_pos):
                    if not self.are_same_pos(self.agent_pos,self.quick_sand_pos):
                        if not self.are_same_pos(self.agent_pos,self.diamond_pos):
                            break
        

        self.create_update_grid()
        observation = np.expand_dims(self.state.copy(),0)

        info = {}

        return observation, info

  
    def take_step(self, action):
        
        self.last_agent_pos  = copy.deepcopy(self.agent_pos)
        
        if action == 0: #down
            self.agent_pos[1] -= 1
        if action == 1: #up
            self.agent_pos[1] += 1
        if action == 2:  #right
            self.agent_pos[0] += 1
        if action == 3: # left
            self.agent_pos[0] -= 1
        
        self.last_action = action
        
        
    
    def get_reward(self):

        reward = -1

        # reward of +10 for getting trophy
        if self.are_same_pos(self.agent_pos, self.goal_pos):
            reward = 10
        
        #reward of -2 if agent tries to go out of bounds
        elif self.are_same_pos(self.agent_pos, self.last_agent_pos):
            reward = -2
        
        #reward of -5 if agent visits quicksand
        elif self.are_same_pos(self.agent_pos,self.quick_sand_pos):
            reward = -5
        
        # reward of +5 if agent picks up diamond
        elif self.diamond_pos is not None and self.are_same_pos(self.agent_pos,self.diamond_pos):
            reward = +5
    
        return reward
    
    
    #overwrite for determinstic and stochastic enviroment
    def environment_type_action(self,action):
        return action
    
    def step(self,action):

        action = self.environment_type_action(action)

        self.take_step(action)
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size[0]-1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size[1]-1)
        
        self.create_update_grid()
        observation = np.expand_dims(self.state.copy(),0)

        #get reward
        reward = self.get_reward()
        self.last_reward = str(reward)

        #update timestep
        self.time_steps += 1

        # Condition to check for termination (episode is over) or reward is attained
        terminated = False
        
        if reward == 10:
            terminated = True
        
        if reward == 5:
            self.diamond_pos = None

        if self.time_steps >= self.max_timesteps:
            terminated = True
        

        # # Condition to check if agent is traversing to a cell beyond the permitted cells
        # # This helps the agent to learn how to behave in a safe and predictable manner
        # truncated = True if np.all((np.asarray(self.agent_pos) >=0 ) & (np.asarray(self.agent_pos) <= 2)) else False

        info = {'action_in_string':self.action_to_string(action)}

        return observation, reward, terminated, False, info
    
    
        
       