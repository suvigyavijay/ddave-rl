from stable_baselines3 import  DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import DangerousDaveEnv
import time, os
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the model")
    argparser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    argparser.add_argument("--model-name", action="store", help="Load the latest model")
    args = argparser.parse_args()

    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "dqn_ddave_{}".format(checkpoint_timestamp)
        
    # Create the DangerousDaveEnv environment
    env = DangerousDaveEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])

    if args.train:
        # Define and train the DQN agent
        model = DQN("CnnPolicy", env, verbose=1, batch_size=64)
        model.learn(total_timesteps=50000, progress_bar=True) 

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
