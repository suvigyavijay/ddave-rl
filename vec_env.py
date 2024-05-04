import gymnasium as gym
from env import DangerousDaveEnv
from ray.rllib.env import VectorEnv
import numpy as np
import pygame
from pygame.locals import QUIT


# Subclass VectorEnv to manage multiple instances of PygameEnv
class VectorizedDDave(VectorEnv):
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [DangerousDaveEnv() for _ in range(num_envs)]
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 60, 1), dtype=np.uint8)
    
    def reset_at(self, index):
        return self.envs[index].reset()
    
    def step_at(self, index, action):
        return self.envs[index].step(action)
    
    def reset(self):
        return [env.reset()[0] for env in self.envs]
    
    def step(self, actions):
        return zip(*[env.step(action) for env, action in zip(self.envs, actions)])

# Example usage
if __name__ == "__main__":
    num_envs = 4
    envs = VectorizedDDave(num_envs)

    obs = envs.reset()
    reward_arr = [0 for _ in range(num_envs)]
    for i in range(1000):
        actions = [envs.action_space.sample() for _ in range(num_envs)]
        obs, rewards, dones, truncs, infos = envs.step(actions)
        reward_arr = [reward + r for reward, r in zip(reward_arr, rewards)]
        print(reward_arr)

    print("Done")
