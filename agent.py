from stable_baselines3 import  DQN,PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.env_checker import check_env
from env import DangerousDaveEnv
import time, os
import argparse
from custom_cnn import policy_kwargs
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the model")
    argparser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    argparser.add_argument("--model-name", action="store", help="Load the latest model")
    argparser.add_argument('--env-rep-type', choices=['text', 'image'])
    argparser.add_argument('--model-type', choices=['DQN','RND','PPO'])
    argparser.add_argument("--retrain", action="store_true", help="Train existing model")
    args = argparser.parse_args()
    print(args)
    
    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "dqn_ddave_{}".format(checkpoint_timestamp)

    if args.env_rep_type:
        env_rep_type = args.env_rep_type
    else:
        env_rep_type = 'image' 
    
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'DQN'
    
    # Create the DangerousDaveEnv environment
    env = DangerousDaveEnv(render_mode="human",env_rep_type=env_rep_type)
    # env = DummyVecEnv([lambda: env])

    if model_type == 'DQN' :

        if args.train:
            # Define and train the DQN agent

            if args.retrain and args.model_name:
                model = DQN.load("checkpoints/{}".format(model_name))
                model.set_env(env)
            else:
                model = DQN("CnnPolicy", env, verbose=1, batch_size=256,policy_kwargs=policy_kwargs,
                            learning_starts=1000,exploration_fraction=0.5,exploration_final_eps=0.01,device=device,
                            target_update_interval=5000,  buffer_size=100000)

            model.learn(total_timesteps=500000, progress_bar=True)
            
            # Save the trained model if desired
            model.save("checkpoints/{}".format(model_name))
        

        if args.evaluate and args.model_name:
            # Evaluate the trained model
            model = DQN.load("checkpoints/{}".format(model_name))
        
        elif args.evaluate:
            # load latest model
            files = os.listdir("checkpoints")
            files.sort(reverse=True)
            latest_checkpoint = files[0]
            model = DQN.load("checkpoints/{}".format(latest_checkpoint))
        
    elif model_type=='PPO':
        if args.train:
            # Define and train the DQN agent

            if args.retrain and args.model_name:
                model = PPO.load("checkpoints/{}".format(model_name))
                model.set_env(env)
            else:
                model = PPO("CnnPolicy", env, verbose=1, batch_size=256,policy_kwargs=policy_kwargs,
                            device=device)

            model.learn(total_timesteps=100000, progress_bar=True)
            
            # Save the trained model if desired
            model.save("checkpoints/{}".format(model_name))
        

        if args.evaluate and args.model_name:
            # Evaluate the trained model
            model = PPO.load("checkpoints/{}".format(model_name))
        elif args.evaluate:
            # load latest model
            files = os.listdir("checkpoints")
            files.sort(reverse=True)
            latest_checkpoint = files[0]
            model = PPO.load("checkpoints/{}".format(latest_checkpoint))

    if args.evaluate:
        obs,info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, info = model.predict(obs)
            obs, rewards, terminated,truncated,info = env.step(action)
            env.render()
