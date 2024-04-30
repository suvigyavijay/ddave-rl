from agents import *
from envs import *
from utils import *

from config import *
from torch.multiprocessing import Pipe
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from gridworld import BaseGridEnvironment

# from tensorboardX import SummaryWriter

import numpy as np


def main():
  
    train_method = default_config['TrainMethod']
    env_id = 'GW'
    env_type = default_config['EnvType']

   
    env = BaseGridEnvironment((4,4),15)

    output_size = env.action_space.n  

   
    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)
    if not os.path.exists('models/'):
        os.makedirs('models/',exist_ok=True)
    
    # writer = SummaryWriter()

    use_cuda = default_config.get('UseGPU')
    use_gae = default_config.get('UseGAE')
    use_noisy_net = default_config.get('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.get('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.get('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 4, 4))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)
  
    agent = RNDAgent
    env_type = DaveEnvironment


    input_size = (1,4,4)
    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )
    
    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')




    final_env = GridWorldEnvironment(env_id=env_id,is_render=is_render,env_idx=0,env=env,sticky_action=sticky_action,p=action_prob,
                         life_done=False,h=4,w=4,history_size=1)

   
    states = final_env.reset()

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []

   
    for step in range(num_step * pre_obs_norm_step):
        action = np.random.randint(0, output_size)
        s, r, d, rd, lr = final_env.run(action)
        # import pdb;pdb.set_trace()
        next_obs.append(s[0, :, :,:].reshape([1, 4,4]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')
    _ = final_env.reset()
    while True:
   
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1
        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states))

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []



            s, r, d, rd, lr = final_env.run(actions[0])
            print(r,actions)

            next_states = s
            rewards = r
            dones = d
            real_dones = rd
            log_rewards = lr
            next_obs = s[0,:,:,:]

            # next_states.append(s[3, :, :,:].reshape([1, 11, 19]))
            # rewards.append(r)
            # dones.append(d)
            # real_dones.append(rd)
            # log_rewards.append(lr)
            # next_obs.append(s[3, :, :,:].reshape([1, 11, 19]))

            # next_states = np.stack(next_states)
            # rewards = np.hstack(rewards)
            # dones = np.hstack(dones)
            # real_dones = np.hstack(real_dones)
            # next_obs = np.stack(next_obs)

         
            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)))
            
            # intrinsic_reward = np.hstack(intrinsic_reward)
            # sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states

            # sample_rall += log_rewards[sample_env_idx]

            # sample_step += 1
            # if real_dones[sample_env_idx]:
            #     sample_episode += 1
            #     # writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
            #     # writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
            #     # writer.add_scalar('data/step', sample_step, sample_episode)
            #     sample_rall = 0
            #     sample_step = 0
            #     sample_i_rall = 0

        
        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states))
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------
        
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 4, 4])
        total_reward = np.stack(total_reward).transpose()
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs)
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        
        total_int_reward = np.stack(total_int_reward).transpose().reshape([-1])
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T]).reshape([-1])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # # -------------------------------------------------------------------------------------------

        # # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)
        
        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model(np.float32(total_state), ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            print(f'Mean Reward {np.mean(total_reward)}')
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)

            eps_reward = []
            for i in range(2):
                eval_env = BaseGridEnvironment((4,4),15)
                eval_obs, eval_info = eval_env.reset()
                eval_terminated = False
                eval_truncated = False
                eval_reward = 0
                while not (eval_terminated or eval_truncated):
                    with torch.no_grad():
                        action, _,_,_ = agent.get_action(np.expand_dims(eval_obs,0))
                    eval_obs, eval_rewards, eval_terminated, eval_truncated, eval_info = eval_env.step(action[0])
                    eval_reward += eval_rewards
                eps_reward.append(eval_reward)
            print(f'{np.mean(eps_reward)} eval reward mean')
            print(eps_reward)


if __name__ == '__main__':
    main()
