import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from segment_tree import MinSegmentTree, SumSegmentTree
from collections import deque
import random
from typing import Deque, Tuple, Dict, List
import math


class ReplayBuffer:
    """A simple numpy replay buffer for parallel environments."""

    def __init__(self, obs_dim: int, size: int, num_envs: int = 1, batch_size: int = 32, n_step: int = 1, gamma: float = 0.99):
        self.obs_buf = np.zeros([size, num_envs, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, num_envs, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, num_envs], dtype=np.float32)
        self.rews_buf = np.zeros([size, num_envs], dtype=np.float32)
        self.done_buf = np.zeros([size, num_envs], dtype=np.float32)
        self.max_size, self.batch_size, self.num_envs = size, batch_size, num_envs
        self.ptr, self.size, = 0, 0
        print(n_step)
        self.n_step_buffer = [deque(maxlen=n_step) for _ in range(num_envs)]
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: np.ndarray):
        for env_idx in range(self.num_envs):
            transition = (obs[env_idx], act[env_idx], rew[env_idx], next_obs[env_idx], done[env_idx])
            self.n_step_buffer[env_idx].append(transition)

            if len(self.n_step_buffer[env_idx]) < self.n_step:
                continue

            # Compute n-step values
            n_rew, n_next_obs, n_done = self._get_n_step_info(self.n_step_buffer[env_idx], self.gamma)
            n_obs, n_act = self.n_step_buffer[env_idx][0][:2]

            idx = (self.ptr + env_idx) % self.max_size
            self.obs_buf[idx, env_idx] = n_obs
            self.next_obs_buf[idx, env_idx] = n_next_obs
            self.acts_buf[idx, env_idx] = n_act
            self.rews_buf[idx, env_idx] = n_rew
            self.done_buf[idx, env_idx] = n_done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float):
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return rew, next_obs, done

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        env_idxs = np.random.choice(self.num_envs, size=self.batch_size, replace=True)
        return {
            'obs': self.obs_buf[idxs, env_idxs],
            'next_obs': self.next_obs_buf[idxs, env_idxs],
            'acts': self.acts_buf[idxs, env_idxs],
            'rews': self.rews_buf[idxs, env_idxs],
            'done': self.done_buf[idxs, env_idxs],
        }

    def __len__(self):
        return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay Buffer for parallel environments."""
    
    def __init__(self, obs_dim: int, size: int, num_envs: int = 1, batch_size: int = 32, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.99):
        super().__init__(obs_dim, size, num_envs, batch_size, n_step, gamma)
        assert alpha >= 0
        self.alpha = alpha
        self.max_priority = 1.0
        self.batch_per_env = self.batch_size // num_envs
    

         # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        # Segment trees for each environment
        self.sum_trees = [SumSegmentTree(tree_capacity) for _ in range(num_envs)]
        self.min_trees = [MinSegmentTree(tree_capacity) for _ in range(num_envs)]

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: np.ndarray):
        super().store(obs, act, rew, next_obs, done)
        for env_idx in range(self.num_envs):
            if len(self.n_step_buffer[env_idx]) >= self.n_step:
                self.sum_trees[env_idx][self.ptr] = self.max_priority ** self.alpha
                self.min_trees[env_idx][self.ptr] = self.max_priority ** self.alpha

    def sample_batch(self, beta: float = 0.4):
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices_list = []
        weights_list = []
        envs_list = []
        for env_idx in range(self.num_envs):
            indices = self._sample_proportional(env_idx)
            envs_list.extend([env_idx for i in indices])
            indices_list.extend(indices)
            weights = np.array([self._calculate_weight(i, beta, env_idx) for i in indices])
            weights_list.append(weights)
        
        indices_np = np.array(indices_list)
        envs_np = np.array(envs_list)

        weights_np = np.concatenate(weights_list,axis=0)
        samples  = {
            'obs':  self.obs_buf[indices_np,envs_np],
        'next_obs' : self.next_obs_buf[indices_np,envs_np],
        'acts' : self.acts_buf[indices_np,envs_np],
        'rews' : self.rews_buf[indices_np,envs_np],
        'done' : self.done_buf[indices_np,envs_np]}

        
        
        return indices_np,envs_np,weights_np, samples
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray, envs_list: List[int]):
        assert len(indices) == len(priorities)
        for idx,envs, priority in zip(indices, envs_list,priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self.sum_trees[envs][idx] = priority ** self.alpha
            self.min_trees[envs][idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, env_idx: int) -> List[int]:
        total_priority = self.sum_trees[env_idx].sum(0, len(self) - 1)
        segment = total_priority / self.batch_per_env
        indices = []
        for _ in range(self.batch_per_env):
            a = segment * _
            b = segment * (_ + 1)
            sample = random.uniform(a, b)
            idx = self.sum_trees[env_idx].retrieve(sample)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx: int, beta: float, env_idx: int):
        p_min = self.min_trees[env_idx].min() / self.sum_trees[env_idx].sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self.sum_trees[env_idx][idx] / self.sum_trees[env_idx].sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight



class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


# ## Rainbow Agent
# 
# Here is a summary of DQNAgent class.
# 
# | Method           | Note                                                 |
# | ---              | ---                                                  |
# |select_action     | select an action from the input state.               |
# |step              | take an action and return the response of the env.   |
# |compute_dqn_loss  | return dqn loss.                                     |
# |update_model      | update the model by gradient descent.                |
# |target_hard_update| hard update from the local model to the target model.|
# |train             | train the agent during num_frames.                   |
# |test              | test the agent (1 episode).                          |
# |plot              | plot the training progresses.                        |
# 
# #### Categorical DQN + Double DQN
# 
# The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation. Here, we use `self.dqn` instead of `self.dqn_target` to obtain the target actions.
# 
# ```
#         # Categorical DQN + Double DQN
#         # target_dqn is used when we don't employ double DQN
#         next_action = self.dqn(next_state).argmax(1)
#         next_dist = self.dqn_target.dist(next_state)
#         next_dist = next_dist[range(self.batch_size), next_action]
# ```


class DQNAgent:
    """DQN Agent interacting with parallel environments.
    
    Attributes:
        env (gym.Env): OpenAI Gym environment or similar that supports batch operations.
        memory (PrioritizedReplayBuffer): Replay memory to store transitions.
        batch_size (int): Batch size for sampling.
        target_update (int): Period for target model's hard update.
        gamma (float): Discount factor.
        dqn (Network): Model to train and select actions.
        dqn_target (Network): Target model to update.
        optimizer (torch.optim): Optimizer for training DQN.
        support (torch.Tensor): Support for categorical DQN.
        use_n_step (bool): Whether to use n_step memory.
        n_step (int): Step number to calculate n-step TD error.
        memory_n (ReplayBuffer): N-step replay buffer.
        device (torch.device): Device to run the model computation.
    """

    def __init__(
        self, 
        env,
        num_env,
        eval_env,
        observation_space_dim,
        action_space_dim,
        memory_size,
        batch_size,
        target_update,
        seed,
        learning_rate,
        gamma=0.99,
        alpha=0.2,
        beta=0.6,
        prior_eps=1e-6,
        v_min=0.0,
        v_max=200.0,
        atom_size=51,
        n_step=3,
        device=None
    ):
        """Initialization with configuration for parallel environments."""
        self.observation_space_dim = observation_space_dim
        self.action_space_dim  = action_space_dim
        self.num_env = num_env
        self.eval_env = eval_env
       
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.seed = seed

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        
        # Initialize memory for 1-step learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.use_n_step = n_step > 1
        self.n_step = n_step
        self.memory =  PrioritizedReplayBuffer(
            obs_dim=self.observation_space_dim,size=memory_size,num_envs=self.num_env,batch_size=self.batch_size,alpha=alpha,n_step=self.n_step,
            gamma=self.gamma)
        
       
        # Initialize memory for N-step learning
        self.use_n_step = n_step > 1
        self.n_step = n_step
        self.memory_n = ReplayBuffer(obs_dim=self.observation_space_dim,size=memory_size,num_envs=self.num_env,
                            batch_size=self.batch_size,n_step=self.n_step,gamma=self.gamma) if self.use_n_step else None
        
        # Setup support for categorical DQN
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
        
        # Initialize networks
        self.dqn = Network(self.observation_space_dim, self.action_space_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(self.observation_space_dim, self.action_space_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=learning_rate)

        # Setup for handling batch operations
        self.is_test = False

    def select_action(self, states: np.ndarray) -> np.ndarray:
        """Select actions for each state in the batch."""
        state_tensor = torch.FloatTensor(states).to(self.device)
        selected_actions = self.dqn(state_tensor).argmax(dim=1).detach().cpu().numpy()
        return selected_actions

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute a batch of actions and observe the results."""
        next_states, rewards, dones, _ = self.env.step(actions)
        return next_states, rewards, dones

    def store_transitions(self, states, actions, rewards, next_states, dones):
        """Store batch of transitions in memory."""
        transition = [states, actions, rewards, next_states, dones]
        self.memory.store(*transition)
        if self.use_n_step:
            self.memory_n.store(*transition)

    def update_model(self) -> float:
        """Update the model by sampling from the memory and performing gradient descent."""
        
        if len(self.memory) < self.batch_size:
            return  

        indices, envs_list,weights, samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(weights).to(self.device)

        # Calculate 1-step learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

      
        loss = torch.mean(elementwise_loss * weights)

        # If using N-step learning, calculate and add n-step loss
        if self.use_n_step:
            gamma_n = self.gamma ** self.n_step
            samples_n = self.memory_n.sample_batch()
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma_n)
            loss += torch.mean(elementwise_loss_n)

        # Calculate weighted loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities,envs_list)

  
        # Reset noise in the networks
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def _target_hard_update(self):
        """Periodically update the target network by copying weights from the primary network."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)


        action_numpy = action.cpu().numpy()
        unique, counts = np.unique(action_numpy, return_counts=True)

        # print(np.asarray((unique, counts)).T)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(next_action.shape[0]), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (next_action.shape[0] - 1) * self.atom_size, next_action.shape[0]
                ).long()
                .unsqueeze(1)
                .expand(next_action.shape[0], self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(next_action.shape[0]), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    

    def learn(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
    
        self.is_test = False
        states = self.env.reset()
        total_loss = []

        for frame_idx in range(1, num_frames + 1):
            
            actions = self.select_action(states)
            next_states, rewards, dones = self.step(actions)
            self.store_transitions(states, actions, rewards, next_states, dones)

            states = next_states

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                total_loss.append(loss)

            if frame_idx % self.target_update == 0:
                self._target_hard_update()
            
            if frame_idx % 2500 == 0:
                print(f'Steps: {frame_idx}')

            if frame_idx % 20000==0:
                print(f'Steps: {frame_idx}')
                print(self.evaluate())
    

    def save(self,path):
        torch.save(self.dqn.state_dict(), path+'.pth')
    
    def load(self,path):
        self.dqn.load_state_dict(torch.load(path+'.pth'))
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
    
    def evaluate(self,num_episodes=1):
        print('Evaluating Result')
        total_rewards = []
        for _ in range(num_episodes):
            obs,info = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            while not (terminated or truncated):
                action  = self.select_action(obs)  # Modify the step function to include an 'explore' parameter
                obs, reward, terminated,truncated, _ = self.eval_env.step(action[0])
                episode_reward += reward
            total_rewards.append(episode_reward)
        average_reward = np.mean(total_rewards)
        print('Evaluating Result',total_rewards)
        return average_reward


    
    
    
        


