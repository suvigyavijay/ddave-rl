# Reinforcement Learning for Dangerous Dave

This repository contains an implementation of reinforcement learning algorithms applied to the classic side-scrolling platform game Dangerous Dave. The game is a Python clone of the original game, which can be found at [mwolfart/dangerous-dave](https://github.com/mwolfart/dangerous-dave).

## Introduction

Dangerous Dave is a challenging side-scrolling platform game where the player controls the character Dave as he navigates through levels filled with enemies and obstacles. In this project, we aim to apply reinforcement learning techniques to train an agent to play Dangerous Dave effectively.

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

## Structure

- `agent.py`: Contains the implementation of the reinforcement learning agent.
- `model.py`: Defines the neural network architecture used by the agent.
- `env.py`: Defines the environment for the Dangerous Dave game.
- `game.py`: Contains the main game loop for the Dangerous Dave game.
- `ddave/`: Contains the game assets and logic for Dangerous Dave.

## Acknowledgments and References

- [mwolfart](https://github.com/mwolfart) for the original Dangerous Dave Python clone.
- [Gymnasium](https://gymnasium.farama.org/index.html) for creating the reinforcement learning environment.
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html) for RL resources.
- [Reinforcement learning with prediction-based rewards](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards) for RND exploration strategy.
- [Proximal Policy Optimization](https://openai.com/index/openai-baselines-ppo) for understanding PPO algorithm.
- [Go-Explore: a New Approach for Hard-Exploration Problems](https://www.uber.com/blog/go-explore/) for inspiration on exploration strategies.
- [CleanRL](https://docs.cleanrl.dev/) for inspiration on PPO and RND implementations.
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) for Vector environment wrapper.
