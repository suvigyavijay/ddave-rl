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

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To test the game, run the following command:
```bash
python ddave/game.py
```

To test the environment, run the following command:
```bash
python env.py
```

To train the reinforcement learning agent, run the following command:
```bash
python agent.py
```

## Reinforcement Learning Algorithms

The repository currently implements the following reinforcement learning algorithms:

- **Deep Q-Network (DQN)**: A deep learning-based reinforcement learning algorithm that uses a neural network to approximate the Q-function.

## Structure

- `train.py`: Script for training the reinforcement learning agent.
- `test.py`: Script for testing the trained agent.
- `agent.py`: Contains the implementation of the reinforcement learning agent.
- `model.py`: Defines the neural network architecture used by the agent.
- `env.py`: Defines the environment for the Dangerous Dave game.
- `utils.py`: Utility functions for preprocessing game states and other helper functions.

## Acknowledgments

- [mwolfart](https://github.com/mwolfart) for the original Dangerous Dave Python clone.
- OpenAI for providing guidance and resources on reinforcement learning techniques.
