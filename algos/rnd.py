
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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import tempfile
from algos.utils import layer_init
from gymnasium.wrappers.normalize import RunningMeanStd

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
            layer_init(nn.Linear(64 * 8 * 4, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 64 * 8 * 4

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RND:
    def __init__(self, envs, eval_env, model_name, total_timesteps=20_000_000, num_steps=1000, num_envs=8, learning_rate=1e-3, 
                 gamma=0.999, gae_lambda=0.95, clip_coef=0.1, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5, 
                 update_epochs=4, norm_adv=True, clip_vloss=True, anneal_lr=True, update_proportion=0.25, int_coef=1.0,
                 ext_coef=2.0, int_gamma=0.99, num_iterations_obs_norm_init=5):
        self.envs = envs
        self.eval_env = eval_env
        self.model_name = model_name
        self.total_timesteps = total_timesteps
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
        self.update_proportion = update_proportion
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.num_iterations_obs_norm_init = num_iterations_obs_norm_init
        
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // 4
        
        self.agent = Agent(self.envs).to(device)
        self.rnd_model = RNDModel(4, self.envs.single_action_space.n).to(device)
        self.combined_parameters = list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters())
        self.optimizer = optim.Adam(
            self.combined_parameters,
            lr=self.learning_rate,
            eps=1e-5,
        )
        self.checkpoint_dir = f"checkpoint/{model_name}"
        self.reward_dir = f"rewards/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)
        
    def save_checkpoint(self, name):
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "rnd_model": self.rnd_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.checkpoint_dir, str(name)),
        )
        
    def save_rewards(self, rewards):
        np.save(f"{self.reward_dir}/rewards.npy", rewards)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint["agent"])
        self.rnd_model.load_state_dict(checkpoint["rnd_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    def train(self):
        episode_rewards = []
        
        reward_rms = RunningMeanStd()
        obs_rms = RunningMeanStd(shape=(1, 1, 96, 60))
        discounted_reward = RewardForwardFilter(self.int_gamma)

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        curiosity_rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        ext_values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        int_values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        avg_returns = deque(maxlen=20)

        # start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.envs.reset()[0]).to(device)
        next_done = torch.zeros(self.num_envs).to(device)
        num_updates = self.total_timesteps // self.batch_size

        print("Start to initialize observation normalization parameter.....")
        next_ob = []
        for step in range(self.num_steps * self.num_iterations_obs_norm_init):
            if step % self.num_steps == 0:
                print(f"Step {step} of {self.num_steps * self.num_iterations_obs_norm_init}")
            acs = np.random.randint(0, self.envs.single_action_space.n, size=(self.num_envs,))
            s, r, d, _ = self.envs.step(acs)
            next_ob += s[:, 3, :, :].reshape([-1, 1, 96, 60]).tolist()

            if len(next_ob) % (self.num_steps * self.num_envs) == 0:
                next_ob = np.stack(next_ob)
                obs_rms.update(next_ob)
                next_ob = []
        print("End to initialize...")

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # action logic
                with torch.no_grad():
                    value_ext, value_int = self.agent.get_value(obs[step])
                    ext_values[step], int_values[step] = (
                        value_ext.flatten(),
                        value_int.flatten(),
                    )
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])

                actions[step] = action
                logprobs[step] = logprob

                # execute the game and log data.
                next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                rnd_next_obs = (
                    (
                        (next_obs[:, 3, :, :].reshape(self.num_envs, 1, 96, 60) - torch.from_numpy(obs_rms.mean).to(device))
                        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                    ).clip(-5, 5)
                ).float()
                target_next_feature = self.rnd_model.target(rnd_next_obs)
                predict_next_feature = self.rnd_model.predictor(rnd_next_obs)
                curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
                for idx, d in enumerate(done):
                    if d:
                        avg_returns.append(self.envs.returned_episode_returns[idx])
                        epi_ret = np.average(avg_returns)
                        print(
                            f"global_step={global_step}, episodic_return={self.envs.returned_episode_returns[idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                        )
                        episode_rewards.append(self.envs.returned_episode_returns[idx])

            curiosity_reward_per_env = np.array(
                [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
            )
            mean, std, count = (
                np.mean(curiosity_reward_per_env),
                np.std(curiosity_reward_per_env),
                len(curiosity_reward_per_env),
            )
            reward_rms.update_from_moments(mean, std**2, count)

            curiosity_rewards /= np.sqrt(reward_rms.var)

            # bootstrap value if not done
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
                ext_advantages = torch.zeros_like(rewards, device=device)
                int_advantages = torch.zeros_like(curiosity_rewards, device=device)
                ext_lastgaelam = 0
                int_lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        ext_nextnonterminal = 1.0 - next_done
                        int_nextnonterminal = 1.0
                        ext_nextvalues = next_value_ext
                        int_nextvalues = next_value_int
                    else:
                        ext_nextnonterminal = 1.0 - dones[t + 1]
                        int_nextnonterminal = 1.0
                        ext_nextvalues = ext_values[t + 1]
                        int_nextvalues = int_values[t + 1]
                    ext_delta = rewards[t] + self.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                    int_delta = curiosity_rewards[t] + self.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                    ext_advantages[t] = ext_lastgaelam = (
                        ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    )
                    int_advantages[t] = int_lastgaelam = (
                        int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                    )
                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)

            b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef

            obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 96, 60).cpu().numpy())

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)

            rnd_next_obs = (
                (
                    (b_obs[:, 3, :, :].reshape(-1, 1, 96, 60) - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()

            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    predict_next_state_feature, target_next_state_feature = self.rnd_model(rnd_next_obs[mb_inds])
                    forward_loss = F.mse_loss(
                        predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                    ).mean(-1)

                    mask = torch.rand(len(forward_loss), device=device)
                    mask = (mask < self.update_proportion).type(torch.FloatTensor).to(device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(
                        mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                    )
                    _, newlogprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
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
                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if self.clip_vloss:
                        ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                        ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                            new_ext_values - b_ext_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                        ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                        ext_v_loss = 0.5 * ext_v_loss_max.mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                    v_loss = ext_v_loss + int_v_loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef + forward_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.combined_parameters,
                            self.max_grad_norm,
                        )
                    self.optimizer.step()
                    
            # print the training statistics
            print(f"Iteration: {update}")
            print("Steps / Second:", int(global_step / (time.time() - start_time)))
            
            # save the model, rewards and evaluate the model
            if update % 20 == 0:
                print("Saving the model and rewards...")
                self.save_checkpoint(update)
                self.save_rewards(episode_rewards)
                print("Evaluating the model...")
                self.evaluate(update)

            # if avg_returns and np.average(avg_returns) > 360:
            #     print("Early Stopping...")
            #     print("Saving the model and rewards...")
            #     self.save_checkpoint(update)
            #     self.save_rewards(episode_rewards)
            #     print("Evaluating the model...")
            #     self.evaluate(update)
            #     break
                
    def evaluate(self, update):
        episode_reward = 0
        done = False
        
        obs = self.eval_env.reset()
        obs = torch.Tensor(np.repeat(obs, self.num_envs, axis=0)).to(device)
        tmp_frame_dir = tempfile.mkdtemp()
        frame_number = 0
        
        while not done:
            action, _, _, _, _ = self.agent.get_action_and_value(obs)
            obs, reward, done, _ = self.eval_env.step(action.cpu().numpy()[:1])
            obs = torch.Tensor(np.repeat(obs, self.num_envs, axis=0)).to(device)
            episode_reward += reward[0]
            
            # save the current frame
            frame_image = (pygame.display.get_surface())
            pygame.image.save(frame_image, f"{tmp_frame_dir}/frame_{frame_number:05d}.png")
            
            frame_number += 1
            
        print(f"Reward: {episode_reward}")
        # create a video from the frames
        os.system(f"ffmpeg -framerate 60 -i {tmp_frame_dir}/frame_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {self.reward_dir}/episode_{update}_{episode_reward}.mp4")
        
        shutil.rmtree(tmp_frame_dir)

