from stable_baselines3 import  DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import DangerousDaveEnv
import time, os
import argparse
from RnD.agents import RNDAgent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the model")
    argparser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    argparser.add_argument("--model-name", action="store", help="Load the latest model")
    argparser.add_argument('--env-rep-type', choices=['text', 'image'])
    argparser.add_argument('--model-type', choices=['DQN','RND'])
    args = argparser.parse_args()
    
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
        model_type = 'DQN'
    else:
        model_type = args.model_type
    
  
    # Create the DangerousDaveEnv environment
    env = DangerousDaveEnv(render_mode="human",env_rep_type=env_rep_type)
    env = DummyVecEnv([lambda: env])


    if model_type == 'DQN':

        if args.train:
            # Define and train the DQN agent
    
            if env_rep_type == 'image':
                model = DQN("CnnPolicy", env, verbose=1, batch_size=64)

            elif env_rep_type == 'text':
                model = DQN("MlpPolicy", env, verbose=1, batch_size=64)

            model.learn(total_timesteps=50, progress_bar=True)
            print(env.envs[0].unqiue_set) 

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

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
