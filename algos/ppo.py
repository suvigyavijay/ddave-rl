import os, shutil
import configparser
import time
from collections import deque
from dataclasses import dataclass
import pygame
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import tempfile
from algos.utils import layer_init

config = configparser.ConfigParser()
config.read('algo.cfg')

torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    
class PPO:
    def __init__(self, envs, eval_env, model_name, total_timesteps=10000000, num_steps=1000, num_envs=8, learning_rate=1e-3, 
                 gamma=0.99, gae_lambda=0.95, clip_coef=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, 
                 update_epochs=4, norm_adv=True, clip_vloss=True, anneal_lr=True):
        self.envs = envs
        self.eval_env = eval_env
        self.model_name = model_name
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.anneal_lr = anneal_lr
        
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // 4
        self.num_iterations = total_timesteps // self.batch_size
        
        self.agent = Agent(envs).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.checkpoint_dir = f"checkpoint/{model_name}"
        self.reward_dir = f"rewards/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)
        
    def save_checkpoint(self, iteration):
        torch.save(self.agent.state_dict(), f"{self.checkpoint_dir}/model_{iteration}.pt")
        
    def save_rewards(self, rewards):
        np.save(f"{self.reward_dir}/rewards.npy", rewards)
        
    def load_checkpoint(self, iteration=None):
        # if iteration is not provided, load the latest model
        if iteration is None:
            iteration = max([int(f.split("_")[1].split(".")[0]) for f in os.listdir(self.checkpoint_dir)])
        self.agent.load_state_dict(torch.load(f"{self.checkpoint_dir}/model_{iteration}.pt"))
        
    def train(self):
        episode_rewards = []
        
        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        avg_returns = deque(maxlen=20)

        # start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.envs.reset()).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # execute the game and log data.
                next_obs, reward, next_done, info = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                for idx, d in enumerate(next_done):
                    if d:
                        avg_returns.append(self.envs.returned_episode_returns[idx])
                        print(f"global_step={global_step}, episodic_return={self.envs.returned_episode_returns[idx]}")
                        episode_rewards.append(self.envs.returned_episode_returns[idx])

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # print the training statistics
            print(f"Iteration: {iteration}")
            print("Steps / Second:", int(global_step / (time.time() - start_time)))
            
            # save the model, rewards and evaluate the model
            if iteration % 20 == 0:
                print("Saving the model and rewards...")
                self.save_checkpoint(iteration)
                self.save_rewards(episode_rewards)
                print("Evaluating the model...")
                self.evaluate(iteration)

            if avg_returns and np.average(avg_returns) > 360:
                print("Early Stopping...")
                print("Saving the model and rewards...")
                self.save_checkpoint(iteration)
                self.save_rewards(episode_rewards)
                print("Evaluating the model...")
                self.evaluate(iteration)
                break
                
    def evaluate(self, iteration):
        episode_reward = 0
        done = False
        
        obs = self.eval_env.reset()
        obs = torch.Tensor(np.repeat(obs, self.num_envs, axis=0)).to(device)
        tmp_frame_dir = tempfile.mkdtemp()
        frame_number = 0
        
        while not done:
            action, _, _, _ = self.agent.get_action_and_value(obs)
            obs, reward, done, _ = self.eval_env.step(action.cpu().numpy()[:1])
            obs = torch.Tensor(np.repeat(obs, self.num_envs, axis=0)).to(device)
            episode_reward += reward[0]
            
            # save the current frame
            frame_image = (pygame.display.get_surface())
            pygame.image.save(frame_image, f"{tmp_frame_dir}/frame_{frame_number:05d}.png")
            
            frame_number += 1
            
        print(f"Reward: {episode_reward}")
        # create a video from the frames
        os.system(f"ffmpeg -framerate 60 -i {tmp_frame_dir}/frame_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {self.reward_dir}/episode_{iteration}_{episode_reward}.mp4")
        
        shutil.rmtree(tmp_frame_dir)
            

