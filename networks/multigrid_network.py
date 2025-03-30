import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_DIRECTIONS = 4

class MultiGridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor using CNN for image and MLP for direction.
    Concatenates the features.
    taken from the original (now commented out) MultiGridNetwork class.

    :param observation_space: The Dict observation space (must contain 'image', 'direction').
    :param features_dim: Output dimension (calculated automatically).
    :param fc_direction: Output dimension for the direction MLP.
    :param kernel_size: Kernel size for CNN layers.
    """
    def __init__(self, observation_space: gym.spaces.Dict,
                 fc_direction: int = 8, kernel_size: int = 3):

        # --- Image Processing Layers --- 
        if 'image' not in observation_space.spaces:
            raise ValueError("Observation space must contain an 'image' key.")
        image_space = observation_space.spaces['image']
        if not isinstance(image_space, gym.spaces.Box):
            raise ValueError(f"Expected 'image' space to be Box, got {type(image_space)}")
        
        # Determine input channels (channels-first or channels-last)
        if len(image_space.shape) == 3:
            if image_space.shape[-1] in [1, 3]: c, h, w = image_space.shape[2], image_space.shape[0], image_space.shape[1]
            else: c, h, w = image_space.shape
        else: raise ValueError(f"Unexpected image space shape: {image_space.shape}")

        # Define CNN architecture (Conv -> ReLU -> Conv -> ReLU -> Flatten)
        cnn_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=kernel_size), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size), nn.LeakyReLU(),
            nn.Flatten(),
        )
        # Calculate flattened CNN output size
        with torch.no_grad():
            cnn_output_size = cnn_layers(torch.zeros(1, c, h, w)).shape[1]
            
        # Final FC layer for image features
        image_fc_output_dim = 64
        self.image_cnn = cnn_layers
        self.image_fc = nn.Sequential(nn.Linear(cnn_output_size, image_fc_output_dim), nn.LeakyReLU())

        # --- Direction Processing Layers --- 
        if 'direction' not in observation_space.spaces:
            raise ValueError("Observation space must contain a 'direction' key.")
        # Direction MLP (takes one-hot encoded direction)
        direction_input_dim = NUM_DIRECTIONS
        direction_mlp_output_dim = fc_direction
        self.direction_mlp = nn.Sequential(nn.Linear(direction_input_dim, direction_mlp_output_dim), nn.ReLU())

        # --- Total Feature Dimension & Super Init --- 
        total_features_dim = image_fc_output_dim + direction_mlp_output_dim
        super().__init__(observation_space, features_dim=total_features_dim)

        # Store config params if needed later (optional)
        self.kernel_size = kernel_size 
        self.fc_direction = fc_direction


    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        :param observations: The input observations, a dictionary of tensors.
        :return: The extracted features tensor.
        """
        # Process Image
        image_obs = observations['image'].to(self.device)
        if image_obs.dim() == 4 and image_obs.shape[-1] in [1, 3]: image_obs = image_obs.permute(0, 3, 1, 2)
        elif image_obs.dim() == 3: # Handle missing batch dim
            if image_obs.shape[-1] in [1, 3]: image_obs = image_obs.permute(2, 0, 1).unsqueeze(0)
            else: image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.float() / 255.0 # Normalize
        image_features = self.image_fc(self.image_cnn(image_obs))

        # Process Direction
        direction_obs = observations['direction'].to(self.device)
        if direction_obs.dim() > 1: direction_obs = direction_obs.squeeze(-1)
        dirs_onehot = torch.nn.functional.one_hot(direction_obs.long(), num_classes=NUM_DIRECTIONS).float()
        direction_features = self.direction_mlp(dirs_onehot)

        # Concatenate features
        combined_features = torch.cat([image_features, direction_features], dim=1)
        return combined_features


class CustomMultiGridPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy using MultiGridFeaturesExtractor.

    :param observation_space: Observation space (gym.spaces.Dict)
    :param action_space: Action space (gym.spaces.Space)
    :param lr_schedule: Learning rate schedule (function that takes current progress remaining as input)
    :param net_arch: Network architecture for policy and value functions. Specifying
                     dict(pi=[...], vf=[...]) is recommended. Default shown below.
    :param activation_fn: Activation function for network layers.
    :param features_extractor_class: Custom features extractor class to use.
    :param features_extractor_kwargs: Keyword arguments to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not (disables automatic normalization).
    :param optimizer_class: The optimizer to use.
    :param optimizer_kwargs: Additional keyword arguments for the optimizer.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        features_extractor_class=MultiGridFeaturesExtractor,
        features_extractor_kwargs=None,
        normalize_images=False, # Normalization is done in extractor
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        **kwargs
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {'fc_direction': 8, 'kernel_size': 3} # Default kwargs for extractor

        if net_arch is None:
            net_arch = dict(pi=[64], vf=[64]) # Default MLP head architecture

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )

'''
class MultiGridNetwork(nn.Module):
    def __init__(self, obs, config, n_actions, n_agents, agent_id):
        super(MultiGridNetwork, self).__init__()
        self.obs_shape = obs
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.agent_id = agent_id

        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),  # [B, 64, 1, 1] -> [B, 64]
            nn.Linear(64, 64),  
            nn.LeakyReLU()
            )

        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS * self.n_agents, self.config.fc_direction),
            nn.ReLU(),
            )

        #interm = (obs['image'].shape[1]-self.config.kernel_size)+1
        self.head = nn.Sequential(
            nn.Linear(64 + self.config.fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        )

    def process_image(self, x):
        if len(x.shape) == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
            
        # Change from (B,H,W,C) to (B,C,W,H) (i.e. RGB channel of dim 3 comes first)
        x = x.permute((0, 3, 1, 2))
        x = x.float()
        return x
            
    def forward(self, obs):
        # process image
        x = torch.tensor(obs['image']).to(device)
        x = self.process_image(x)
        batch_dim = x.shape[0]

        # Run conv layers on image
        image_features = self.image_layers(x)
        image_features = image_features.reshape(batch_dim, -1)

        # Process direction and run direction layers
        dirs = torch.tensor(obs['direction']).to(device)
        if batch_dim == 1:  # 
            dirs = torch.tensor(dirs).unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=NUM_DIRECTIONS).reshape((batch_dim, -1)).float()
        dirs_encoding = self.direction_layers(dirs_onehot)

        # Concat
        features = torch.cat([image_features, dirs_encoding], dim=-1)

        # Run head
        return self.head(features)
'''