import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIRECTIONS = 4

class MultiGridNetwork(nn.Module):
    def __init__(self, obs, config, n_actions, n_agents, agent_id):
        """
        Initialize Deep Q Network
        """
        super(MultiGridNetwork, self).__init__()
        self.obs_shape = obs
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.agent_id = agent_id
        
        # Simpler fixed architecture like the professor's code
        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, (config.kernel_size, config.kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (config.kernel_size, config.kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
            
        # Direction encoding remains fixed
        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, config.fc_direction),
            nn.ReLU(),
        )
        
        # Simpler head
        self.head = nn.Sequential(
            nn.Linear(64 + config.fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        )
        
        # Move to device
        self.to(device)

    def process_image(self, x):
        """Process image observations into the proper tensor format for convolution"""
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        
        # Make sure dimensions and type are correct
        x = x.float()
        
        # Handle dimension ordering
        if len(x.shape) == 3:  # [H,W,C]
            x = x.permute(2, 0, 1).unsqueeze(0)  # -> [1,C,H,W]
        elif len(x.shape) == 4 and x.shape[3] == 3:  # [B,H,W,C]
            x = x.permute(0, 3, 1, 2)  # -> [B,C,H,W]
            
        return x
            
    def forward(self, obs):
        # Process image
        if isinstance(obs, dict) and 'image' in obs:
            x = obs['image']
            if isinstance(x, list) or isinstance(x, np.ndarray):
                x = torch.tensor(x).to(device)
        else:
            return torch.zeros(self.n_actions).to(device)
            
        x = self.process_image(x)
        batch_dim = x.shape[0]
        
        # Run conv layers on image
        image_features = self.image_layers(x)
        image_features = image_features.reshape(batch_dim, -1)
        
        # Process direction and run direction layers
        if isinstance(obs, dict) and 'direction' in obs:
            dirs = obs['direction']
            if isinstance(dirs, list) or isinstance(dirs, np.ndarray):
                dirs = torch.tensor(dirs).to(device)
            if batch_dim == 1 and not isinstance(dirs, torch.Tensor):
                dirs = torch.tensor(dirs)
                dirs = dirs.unsqueeze(0)
            elif batch_dim == 1 and len(dirs.shape) == 0:
                dirs = dirs.unsqueeze(0)
            dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=NUM_DIRECTIONS).float()
            dirs_onehot = dirs_onehot.reshape(batch_dim, -1).to(device)
            dirs_encoding = self.direction_layers(dirs_onehot)
        else:
            dirs_encoding = torch.zeros(batch_dim, self.config.fc_direction).to(device)

        # Concat features
        features = torch.cat([image_features, dirs_encoding], dim=-1)

        # Run head
        return self.head(features)