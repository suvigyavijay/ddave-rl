import numpy as np
import gymnasium as gym
import torch

from gymnasium.core import Wrapper
from typing import Any

class WrapperMod(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        """Ignore the seed and options arguments."""
        return self.env.reset()

class RecordEpisodeStatistics(WrapperMod):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - np.asarray(dones, dtype=np.int32)
        self.episode_lengths *= 1 - np.asarray(dones, dtype=np.int32)
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
