import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print
from env import DangerousDaveEnv
import time
import os
import argparse

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

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

    model_func = DQNConfig if args.model_type == "dqn" else PPOConfig

    algo = (
        model_func()
        .rollouts(num_rollout_workers=32)
        .resources(num_gpus=1)
        .environment(env=DangerousDaveEnv)
        .training(
            model=dict(
                conv_filters=[
                    # input image 96x60x1
                    [32, [4, 3], 6], 
                    [64, [4, 3], 4], 
                    [128, [4, 3], 4], 
                ],
            ),
        )
        .update_from_dict(
            {
                "gamma": 0.99,
                "train_batch_size": 128,
                # "num_atoms": 32,
                # "noisy": True,
                # "lr": 0.01,
                # "hiddens": [512],
                # "target_network_update_freq": 10,
                # "replay_buffer_config": {
                #     "prioritized_replay_alpha": 0.5,
                #     "capacity": 50000,
                # },
                # "num_steps_sampled_before_learning_starts": 1000,
                # "n_step": 1,
                # "gpu": True,
                # "v_min": -1000.0,
                # "v_max": 1000.0,
            }
        )
        .build()
    )

    if args.train:
        # Configure and train the DQN agent using Ray RLLib

        # Save the trained model if desired
        for i in range(100):
            print("Training iteration: ", i)
            result = algo.train()
            print(pretty_print(result))
            if i % 10 == 0:
                algo.save("checkpoints/{}".format(model_name))


    if args.evaluate: 
        env = DangerousDaveEnv()

        # get the latest checkpoint
        models = os.listdir("checkpoints/")
        models.sort(reverse=True)
        model_name = models[0]
        print("Loading model: ", model_name)
        # algo = DQNConfig().environment(DangerousDaveEnv).build()
        algo = Algorithm.from_checkpoint("checkpoints/{}".format(model_name))
        algo.reset_config({
            "num_workers": 0,
            "evaluation_num_workers": 1,
            "evaluation_interval": 1,
            "in_evaluation": True,
        })

        episode_reward = 0
        terminated = truncated = False
        obs, info = env.reset()

        # Create a directory to store frames
        if not os.path.exists("frames"):
            os.makedirs("frames")
        
        frame_number = 0
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Save the current frame
            frame_image = (pygame.display.get_surface())
            pygame.image.save(frame_image, f"frames/frame_{frame_number:05d}.png")
            frame_number += 1

        print("Reward: ", episode_reward)

        # Assuming ffmpeg is installed, compile frames into a video
        os.system("ffmpeg -r 60 -f image2 -s 1920x1080 -i frames/frame_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p evaluation_video.mp4")

