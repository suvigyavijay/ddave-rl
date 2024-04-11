import torch as th
import torch.nn as nn
from gymnasium import spaces
import torch


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
   
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 2), stride=1, padding=1)  # Output size: (16, 11, 100)
        self.pool = nn.MaxPool2d(kernel_size=(1, 1), stride=2)  # Output size: (16, 5, 50)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 2), stride=1, padding=1)  # Output size: (32, 5, 50)
        # After another pooling layer, the size would be (32, 2, 25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1, padding=1)  # Output size: (64, 2, 25) - No padding
        # Flattening the output for the fully connected layer
        self.fc1 = nn.Linear(19712, 1024)
        self.fc2 = nn.Linear(1024, features_dim)  # Fully connected layer to output 128 features

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        x = th.relu(self.conv3(x))
        
        # Flatten the output for dense layer
        x = x.view(-1, 19712)
        x = self.fc1(x)
        x = th.relu(self.fc2(x))
        return x

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
