import ray
from ray.rllib.algorithms.dqn import DQNConfig
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
    args = argparser.parse_args()

    checkpoint_timestamp = int(time.time())

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "dqn_ddave_{}".format(checkpoint_timestamp)

    if args.train:
        # Configure and train the DQN agent using Ray RLLib
        algo = (
            DQNConfig()
            .rollouts(num_rollout_workers=8)
            .resources(num_gpus=0)
            .environment(env=DangerousDaveEnv)
            .training(
                model=dict(
                    conv_filters=[[32, [4, 3], 2], [64, [4, 3], 2], [128, [4, 3], 2], [256, [4, 3], 2]],
                ),
                train_batch_size=32,
                gamma=0.99,
            )
            # .update_from_dict(
            #     {
            #         "replay_buffer_config": {
            #             "capacity": 1000000,
            #         },
            #     }
            # )
            .build()
        )

        # Save the trained model if desired
        for i in range(10):
            print("Training iteration: ", i)
            result = algo.train()
            print(pretty_print(result))
        algo.save("checkpoints/{}".format(model_name))

    if args.evaluate: 
        env = DangerousDaveEnv()

        episode_reward = 0
        terminated = truncated = False

        obs, info = env.reset()

        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            env.render()

        print("Reward: ", episode_reward)


