from stable_baselines3 import  DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env_grid import DangerousDaveEnv
import time, os
import argparse
import pygame


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the model")
    argparser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    argparser.add_argument("--model-name", action="store", help="Load the latest model")
    argparser.add_argument("--model-type", action="store", default="dqn", choices=["dqn", "ppo"], help="Choose the model type")
    args = argparser.parse_args()

    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "{}_ddave_{}".format(args.model_type, checkpoint_timestamp)
        
    # Create the DangerousDaveEnv environment
    eval_env = DangerousDaveEnv(render_mode="human", env_rep_type="text")
    env = DummyVecEnv([lambda: eval_env])
    envs = SubprocVecEnv([lambda : DangerousDaveEnv(env_rep_type="text") for _ in range(64)])
    
    model_func = DQN if args.model_type == "dqn" else PPO

    eval_callback = EvalCallback(eval_env, best_model_save_path="./sb3_checkpoint/",
                             log_path="./sb3_checkpoint_logs/", eval_freq=10000,
                             deterministic=True, render=False)

    if args.train:
        # Define and train the DQN agent
        model = model_func("MlpPolicy", envs, verbose=1, batch_size=64)
        model.learn(total_timesteps=10000000, progress_bar=True, callback=eval_callback) 

        # Save the trained model if desired
        model.save("checkpoints/{}".format(model_name))

    if args.evaluate and args.model_name:
        # Evaluate the trained model
        model = model_func.load("checkpoints/{}".format(model_name))
    elif args.evaluate:
        # load latest model
        files = os.listdir("checkpoints")
        files.sort(reverse=True)
        latest_checkpoint = files[0]
        model = model_func.load("checkpoints/{}".format(latest_checkpoint))

        episode_reward = 0
        terminated = truncated = False
        obs = env.reset()

        # Create a directory to store frames
        if not os.path.exists("frames"):
            os.makedirs("frames")
        
        frame_number = 0
        while not terminated:
            action = model.predict(obs)[0]
            obs, reward, terminated, info = env.step(action)
            episode_reward += reward

            # Save the current frame
            frame_image = (pygame.display.get_surface())
            pygame.image.save(frame_image, f"frames/frame_{frame_number:05d}.png")
            frame_number += 1

        print("Reward: ", episode_reward)

        # Assuming ffmpeg is installed, compile frames into a video
        os.remove("evaluation_video.mp4")
        os.system("ffmpeg -r 60 -f image2 -s 1920x1080 -i frames/frame_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p evaluation_video.mp4")
