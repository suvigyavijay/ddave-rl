# Reinforcement Learning for Dangerous Dave

This repository contains an implementation of reinforcement learning algorithms applied to the classic side-scrolling platform game Dangerous Dave. The game is a Python clone of the original game, which can be found at [mwolfart/dangerous-dave](https://github.com/mwolfart/dangerous-dave).

## Introduction

Dangerous Dave is a challenging side-scrolling platform game where the player controls the character Dave as he navigates through levels filled with enemies and obstacles. In this project, we aim to apply reinforcement learning techniques to train an agent to play Dangerous Dave effectively. Drawing inspiration from successful RL techniques used in similar games, such as Montezuma's Revenge, we focus on enhancing exploration and learning efficiency through advanced methods like Random Network Distillation (RND) and Proximal Policy Optimization (PPO).

Here's a demo of the trained agent playing Dangerous Dave:

![Trained Agent Demo](demo.gif)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/suvigyavijay/dangerous-dave-rl.git
    ```

2. Navigate to the project directory:
    ```bash
    cd ddave-rl
    ```

3. Create a virtual environment and install the dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

To test the game, run the following command:
```bash
python game.py
```

To test the environment, run the following command:
```bash
python env.py
```

To train the reinforcement learning agent, run the following command:
```bash
python agent.py --train --model-type ppo
```

## Reinforcement Learning Algorithms

The repository currently implements the following reinforcement learning algorithms:

- **Proximal Policy Optimization (PPO)**: A policy gradient method that aims to optimize the policy in a stable manner by constraining the update step.
- **Random Network Distillation (RND)**: An exploration strategy that uses a random neural network to predict the intrinsic reward signal.

## Configuration Files

- **algo.cfg**: Configuration file for the RL algorithms, specifying parameters such as total timesteps and common settings.
- **game.cfg**: Configuration file for the game environment, specifying details such as map dimensions, reward structures, and termination conditions.

## Folder Structure

- `agent.py`: Contains the implementation of the reinforcement learning agent.
- `algo.cfg`: Configuration file for the RL algorithms.
- `algos/`: Directory containing the implementation of RL algorithms.
  - `ppo.py`: Implementation of the PPO algorithm.
  - `rnd.py`: Implementation of the RND algorithm.
  - `utils.py`: Utility functions for the RL algorithms.
- `ddave/`: Contains the game assets and logic for Dangerous Dave.
  - `__init__.py`: Initializes the `ddave` module.
  - `helper.py`: Helper functions for the game.
  - `levels/`: Directory containing level design files.
    - `1.txt`: Configuration for Level 1.
    - `2.txt`: Configuration for Level 2.
    - `3.txt`: Configuration for Level 3.
  - `tiles/`: Directory containing image resources for the game.
    - `game/`: Subdirectory with image files for game elements.
    - `ui/`: Subdirectory with image files for the user interface.
  - `utils.py`: Additional utility functions specific to Dangerous Dave.
- `env.py`: Defines the environment for the Dangerous Dave game.
- `game.cfg`: Configuration file for the game environment.
- `game.py`: Contains the main game loop for Dangerous Dave.
- `requirements.txt`: Lists the dependencies required to run the project.

## Project Details

### Environment Setup

The game environment is designed with both image-based and text-based observation spaces, allowing various types of observation inputs for reinforcement learning algorithms. Key configurable parameters include trophy score, item score, observation space representation, random agent respawn, grid world configuration, false door episode termination, and current level.

### Training and Vectorized Environment

We used a vectorized environment to run the game and collect samples with different policies. Specifically, we used 64 parallel environments (`NUM_ENVS = 64`) and trained them on a 64-core processor using a customized VecEnv wrapper from stable baselines. This setup allowed us to efficiently collect diverse experiences and significantly speed up the training process. One round of training takes around 3 hours with this configuration. Using this setup, the agent was able to solve Level 1 and began learning Level 2 in Dangerous Dave.

## Acknowledgments and References

- [mwolfart](https://github.com/mwolfart) for the original Dangerous Dave Python clone.
- [Gymnasium](https://gymnasium.farama.org/index.html) for creating the reinforcement learning environment.
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html) for RL resources.
- [Reinforcement learning with prediction-based rewards](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards) for RND exploration strategy.
- [Proximal Policy Optimization](https://openai.com/index/openai-baselines-ppo) for understanding PPO algorithm.
- [Go-Explore: a New Approach for Hard-Exploration Problems](https://www.uber.com/blog/go-explore/) for inspiration on exploration strategies.
- [CleanRL](https://docs.cleanrl.dev/) for inspiration on PPO and RND implementations.
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) for Vector environment wrapper.
