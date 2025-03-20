import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIRECTIONS = 4

class MultiGridNetwork(nn.Module):
    def __init__(self, obs, config, n_actions, n_agents, agent_id):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(MultiGridNetwork, self).__init__()
        self.obs_shape = obs
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.agent_id = agent_id
        
        # We'll initialize the convolutional layers dynamically in the forward pass
        # This makes the network adapt to whatever input it receives
        self.image_layers = None
        self.post_conv = None
        self.feature_size = None
        self.in_channels = None
        self.h = None
        self.w = None
        self.initialized = False
            
        # Direction encoding remains fixed
        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, self.config.fc_direction),
            nn.ReLU(),
        )
        
        # Add a network for processing other agents' actions (for coordinated training)
        self.other_agent_layers = nn.Sequential(
            nn.Linear(n_agents-1, 32),  # Process n_agents-1 actions
            nn.ReLU(),
        )
        
        # Move the direction layers to the device
        self.direction_layers = self.direction_layers.to(device)
        self.other_agent_layers = self.other_agent_layers.to(device)

        # Head will be initialized after we know the feature size
        self.head = None
        
        # Move the entire model to the device
        self.to(device)

    def initialize_layers(self, x):
        """Initialize layers based on actual input dimensions"""
        # Get dimensions from actual input
        batch_size, in_channels, h, w = x.shape
        self.in_channels = in_channels
        self.h = h
        self.w = w
        
        # Use small kernel for small inputs
        self.kernel_size = min(self.config.kernel_size, min(self.h, self.w))
        if min(self.h, self.w) <= 3:
            self.kernel_size = 1
            
        print(f"Initializing MultiGridNetwork with actual input: channels={in_channels}, size={h}x{w}, kernel={self.kernel_size}")
        
        # Compute output size after convolutions
        h_out = self.h - 2 * (self.kernel_size - 1)
        w_out = self.w - 2 * (self.kernel_size - 1)
        
        # Use 1x1 kernels if needed
        if h_out <= 0 or w_out <= 0:
            self.kernel_size = 1
            h_out = self.h
            w_out = self.w
        
        # Initialize convolutional layers
        self.image_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, (self.kernel_size, self.kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (self.kernel_size, self.kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten()
        ).to(device)
        
        # Calculate feature size
        self.feature_size = 64 * h_out * w_out
        
        # Initialize post-conv layers
        self.post_conv = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.LeakyReLU()
        ).to(device)
        
        # Initialize head
        self.head = nn.Sequential(
            nn.Linear(64 + self.config.fc_direction + 32, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        ).to(device)
        print("[multigrid_network] device: " + str(device))
        
        self.initialized = True

    def process_image(self, x):
        """Process image observations into the proper tensor format for convolution
        
        Args:
            x: Image observation tensor or array
            
        Returns:
            Properly formatted tensor for convolution
        """
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        
        # Make sure dimensions and type are correct
        x = x.float()
        
        # Batch dimension handling
        if len(x.shape) == 3:  # Single image: [H,W,C] format
            # Transpose from [H,W,C] to [C,H,W] for PyTorch
            x = x.permute(2, 0, 1).unsqueeze(0)  # Now [1,C,H,W]
        elif len(x.shape) == 4:  # Batched images
            # Check if already in [B,C,H,W] format or if in [B,H,W,C]
            if x.shape[1] == 3 or x.shape[1] == 1:  # Already [B,C,H,W]
                pass  # Already in correct format
            else:  # Likely [B,H,W,C]
                x = x.permute(0, 3, 1, 2)  # Convert to [B,C,H,W]
        else:
            print(f"Warning: unusual image shape: {x.shape}")
            try:
                if len(x.shape) == 2:  # Grayscale image [H,W]
                    # Add channel dimension and batch dimension
                    x = x.unsqueeze(0).unsqueeze(0)  # Now [1,1,H,W]
                else:
                    # Last resort - try to reshape as best we can
                    x = x.reshape(1, 3, 5, 5)  # Force into expected shape
            except Exception as e:
                print(f"Failed to reshape tensor: {e}")
                # Return a dummy tensor if all else fails
                return torch.zeros((1, 3, 5, 5)).to(device)
        
        return x
            
    def forward(self, obs):
        # Process image
        if isinstance(obs, dict) and 'image' in obs:
            x = obs['image']
            # Convert lists to numpy arrays first
            if isinstance(x, list):
                x = np.array(x)
            
            if isinstance(x, np.ndarray):
                x = torch.tensor(x).to(device)
        else:
            print(f"Warning: unexpected observation format: {type(obs)}")
            # Return zeros, but make sure it's shaped correctly for deterministic action selection
            return torch.zeros(7).to(device)  # Only need logits for 7 discrete actions (0-6)
            
        x = self.process_image(x)
        batch_dim = x.shape[0]
        
        # Initialize layers if this is the first forward pass
        if not self.initialized:
            self.initialize_layers(x)
            
        # Run conv layers on image
        conv_features = self.image_layers(x)
        image_features = self.post_conv(conv_features)
        image_features = image_features.reshape(batch_dim, -1)
        
        # Process direction and run direction layers
        if isinstance(obs, dict) and 'direction' in obs:
            dirs = obs['direction']
            if isinstance(dirs, list):
                dirs = np.array(dirs)
            if isinstance(dirs, np.ndarray):
                dirs = torch.tensor(dirs).to(device)
            if batch_dim == 1:
                dirs = torch.tensor(dirs)
                dirs = dirs.unsqueeze(0)
            dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=NUM_DIRECTIONS).float()
            dirs_onehot = dirs_onehot.reshape(batch_dim, -1).to(device)
            dirs_encoding = self.direction_layers(dirs_onehot)
        else:
            dirs_encoding = torch.zeros(batch_dim, self.config.fc_direction).to(device)

        # Process other agents' actions (if available)
        other_agent_encoding = torch.zeros(batch_dim, 32).to(device)
        if isinstance(obs, dict) and 'other_actions' in obs:
            other_actions = obs['other_actions']
            if isinstance(other_actions, list):
                other_actions = np.array(other_actions)
            if isinstance(other_actions, np.ndarray):
                # Explicitly convert to float tensor
                other_actions = torch.tensor(other_actions, dtype=torch.float32).to(device)
            elif isinstance(other_actions, torch.Tensor) and other_actions.dtype != torch.float32:
                # Ensure tensor is float type if it's already a tensor
                other_actions = other_actions.float()
            
            if batch_dim == 1 and len(other_actions.shape) == 1:
                other_actions = other_actions.unsqueeze(0)
            
            other_agent_encoding = self.other_agent_layers(other_actions)

        # Concat all features
        features = torch.cat([image_features, dirs_encoding, other_agent_encoding], dim=-1)

        # Run head to get logits - we'll return raw logits for deterministic action selection
        logits = self.head(features)
        
        return logits