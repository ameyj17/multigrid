from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
from typing import Dict, List, Tuple, Optional, Type, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time 
# Add this import for utils
import utils

from utils import plot_single_frame, make_video, extract_mode_from_path
from networks.multigrid_network import MultiGridNetwork

class DictFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor that passes dict observations directly to the policy network"""
    
    def __init__(self, observation_space, features_dim=1):
        print(f"DictFeaturesExtractor initialized with observation space: {observation_space}")  # Print observation space
        super().__init__(observation_space, features_dim)
        
    def forward(self, observations):
        """Extract tensors from the observations dictionary"""
        print("[DictFeaturesExtractor] observations: ", observations)

        # Extracting the tensors directly
        direction_tensor = observations['direction']  # Shape: (1, 1)
        image_tensor = observations['image']          # Shape: (1, 3, 5, 5)
        other_actions_tensor = observations['other_actions']  # Shape: (1, 2)

        # Ensure direction_tensor has the correct dimensions
        if direction_tensor.dim() == 2:  # If it's (1, 1)
            direction_tensor = direction_tensor.unsqueeze(2).unsqueeze(3)  # Shape: (1, 1, 1, 1)

        # Now expand direction_tensor to match the image_tensor's dimensions
        direction_tensor = direction_tensor.expand(-1, -1, image_tensor.size(2), image_tensor.size(3))  # Shape: (1, 1, 5, 5)

        # Now concatenate along the appropriate dimension (e.g., channels)
        combined_tensor = torch.cat((image_tensor, direction_tensor), dim=1)  # Shape: (1, 4, 5, 5)
        print("[DictFeaturesExctractor] combined_tensor:  " + str(combined_tensor.shape))

        return combined_tensor  # Return the combined tensor

class CustomPolicy(ActorCriticPolicy):
    """Custom policy that uses MultiGridNetwork for both actor and critic"""
    def __init__(self, observation_space, action_space, lr_schedule, config, n_agents, agent_id, **kwargs):
        # Use our custom features extractor
        kwargs["features_extractor_class"] = DictFeaturesExtractor
        
        # Initialize with minimal parameters
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Skip default network architecture since we'll use MultiGridNetwork
            net_arch=None,
            activation_fn=torch.nn.ReLU,
            **kwargs
        )
        # Create actor network (policy)
        self.actor_net = MultiGridNetwork(
            observation_space,
            config,
            int(np.prod(action_space.shape)),  # Changed to match discrete actions
            n_agents,
            agent_id
        )
        
        # Create critic network (value function)
        self.critic_net = MultiGridNetwork(
            observation_space,
            config,
            1,  # Value function has only one output
            n_agents,
            agent_id
        )
        
        # Override the policy forward method to handle dict observations
        self._policy_forward = self._policy_forward_dict
        self.process_observation = self.process_observation_dict
        
    def process_observation_dict(self, obs):
        """Process dictionary observations to ensure they're tensors"""
        return obs

    def _policy_forward_dict(self, obs_dict):
        """Custom forward pass for the policy to handle dictionary observations"""
        return self.actor_net(obs_dict)
    
    def predict_values(self, obs):
        """Override predict_values to use our custom critic network"""
        obs = self.process_observation(obs)
        values = self.critic_net(obs)
        return values
    
    def evaluate_actions(self, obs, actions):
        """
        Override evaluate_actions to use our custom actor and critic networks
        
        Args:
            obs: Observation from the environment
            actions: Actions whose value and log probability we want to compute
            
        Returns:
            values: Value function predictions
            log_probs: Log probabilities of the actions
            entropy: Entropy of the action distribution
        """
        obs = self.process_observation(obs)
        
        # Get action logits from the actor network
        action_logits = self.actor_net(obs)
        
        # Create distribution
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Get log probabilities
        log_prob = distribution.log_prob(actions)
        
        # Get entropy
        entropy = distribution.entropy()
        
        # Get values from critic network
        values = self.critic_net(obs)
        
        return values, log_prob, entropy
    
    def forward(self, obs):
        """Forward pass in all the networks (actor and critic)"""
        obs = self.process_observation(obs)  # Ensure obs is processed

        # Run the observation through the actor network to get action logits
        actor_features = self.actor_net(obs)
        
        # Create categorical distribution directly with logits
        distribution = self.action_dist.proba_distribution(action_logits=actor_features)
        
        # Sample an action
        actions = distribution.get_actions()
        log_probs = distribution.log_prob(actions)
        
        # Get values from critic network
        critic_features = self.critic_net(obs)
        values = critic_features
        
        return actions, values, log_probs
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Override the default predict method to handle our specific action shape"""
        observation = self.process_observation(observation)
        with torch.no_grad():
            action_logits = self.actor_net(observation)
            
            # For deterministic policy, simply take the action with highest logit
            if isinstance(action_logits, torch.Tensor):
                actions = torch.argmax(action_logits[:7]).cpu().numpy()  # Only consider first 7 actions (0-6)
            else:
                # Fallback for unexpected output
                actions = np.array(0)
        
        return actions, None
class MultiAgentToSingleAgentWrapper(gym.Wrapper):
    """Wrapper that converts multi-agent environment to single-agent for one agent."""
    
    def __init__(self, env, agent_id=0):
        super().__init__(env)
        self.agent_id = agent_id
        self.n_agents = getattr(env, 'n_agents', 3)  # Default to 3 if not specified
        
        # Initialize last_actions with zeros
        self.last_actions = [0] * self.n_agents
        
        # Define the action space for this agent
        self.action_space = spaces.Discrete(7)  # MultiGrid has 7 actions
        
        # Set up the observation space for this agent
        if hasattr(env, 'observation_space') and isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space
        else:
            # Default observation space structure for MultiGrid
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(3, 5, 5, 3),  # Channel-first format for SB3
                    dtype=np.uint8
                ),
                'direction': spaces.Box(
                    low=0,
                    high=3,
                    shape=(3,),
                    dtype=np.uint8
                )
            })
        
        # Print spaces for debugging
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")
        print(f"Initialized wrapper for agent {self.agent_id}")
        
    def reset(self):
        """Reset the environment and return the observation for this agent."""
        obs = self.env.reset()
        # Reset last_actions to zeros
        self.last_actions = [0] * self.n_agents
        # Get this agent's observation
        agent_obs = self.get_agent_obs(obs, self.agent_id)
        return agent_obs
    
    def get_agent_obs(self, obs, agent_id):
        """Extract observation for a specific agent from the environment observation."""
        # Handle different observation structures
        if isinstance(obs, dict):
            # If observations are already structured by agent
            if 'image' in obs and isinstance(obs['image'], np.ndarray):
                # If image is a single array for all agents, we need to process it
                if len(obs['image'].shape) == 4 and obs['image'].shape[0] == self.n_agents:
                    # If dimensions are (n_agents, height, width, channels)
                    image = obs['image'][agent_id]
                else:
                    # Otherwise assume the image is already for this agent
                    image = obs['image']
                
                # Convert image from (H, W, C) to (C, H, W) for SB3
                if len(image.shape) == 3 and image.shape[2] == 3:  # If shape is (H, W, C)
                    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
                
                # For direction, ensure it's the right agent's direction
                if 'direction' in obs and isinstance(obs['direction'], np.ndarray):
                    if len(obs['direction'].shape) > 1 and obs['direction'].shape[0] == self.n_agents:
                        direction = obs['direction'][agent_id]
                    else:
                        direction = obs['direction']
                else:
                    direction = np.zeros(3, dtype=np.uint8)  # Default direction
                
                return {
                    'image': image,
                    'direction': direction
                }
        
        # Fallback: return a structured observation with defaults
        return {
            'image': np.zeros((3, 5, 5, 3), dtype=np.uint8),  # Channel-first empty image
            'direction': np.zeros(3, dtype=np.uint8)  # Default direction
        }
    
    def step(self, action):
        """
        Take a step in the environment with the given action for this agent only.
        """
        # Ensure action is an integer in the valid range
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Ensure action is within valid range for MultiGrid (0-6)
        action = int(action) % 7
        
        # Update this agent's action in the full action list
        self.last_actions[self.agent_id] = action
        
        # Execute the step with all actions
        next_obs, rewards, done, info = self.env.step(self.last_actions)
        
        # Get this agent's observation and reward
        agent_obs = self.get_agent_obs(next_obs, self.agent_id)
        
        if isinstance(rewards, (list, tuple)) and len(rewards) > self.agent_id:
            reward = rewards[self.agent_id]
        else:
            reward = rewards
        
        return agent_obs, reward, done, info

class MetaController:
    """Coordinates multiple PPO agents with the multigrid environment"""
    def __init__(self, config, env, device, training=True, debug=False):
        """Initialize metacontroller for MultiGrid environment"""
        self.config = config
        self.debug = debug
        self.device = device
        
        # Inspect environment to understand structure
        if hasattr(env, 'num_agents'):
            self.n_agents = env.num_agents
        elif hasattr(env, 'agents') and isinstance(env.agents, list):
            self.n_agents = len(env.agents)
        else:
            print("Warning: Could not detect number of agents, defaulting to 3")
            self.n_agents = 3
        
        print(f"Environment has {self.n_agents} agents")
        
        # Inspect action space
        if hasattr(env, 'action_space'):
            print(f"Environment action space: {env.action_space}")
        
        # Check if the environment has an Actions enum
        if hasattr(env, 'Actions'):
            print("Environment has Actions enum:")
            for action in env.Actions:
                print(f"  {action.value}: {action.name}")
        
        self.total_steps = 0
        self.training = training
        
        # Store metrics
        self.metrics = {f"agent_{i}_reward": [] for i in range(self.n_agents)}
        self.metrics["episodes"] = []
        
        # Set n_envs from config or default to 16
        self.n_envs = getattr(config, 'n_envs', 16)  # Match config value
        
        # For decentralized training, we need one PPO agent per agent
        self.ppo_agents = []
        self.vec_envs = []
        
        # Decide on decentralized or centralized training
        self.decentralized = getattr(config, 'decentralized_training', True)
        print(f"Training mode: {'Decentralized' if self.decentralized else 'Coordinated'}")
        
        # Set up the model name
        if not hasattr(config, 'model_name'):
            config.model_name = f"{config.mode}_seed_{config.seed}_domain_{config.domain}_version_{config.version}"
        
        # Initialize wandb if not in debug mode
        if not debug and training:
            import wandb
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=config.model_name
            )
        else:
            self.wandb_run = None
        
        # Create the environment and wrappers
        for i in range(self.n_agents):
            print(f"Creating PPO agent {i}")
            
            # Helper function to create environment
            def make_env(env_id=None):
                # Create a new environment instance
                e = utils.make_env(config)
                
                # DEBUG: Print information about the environment
                if hasattr(e, 'action_space'):
                    print(f"Agent {i} - New env action space: {e.action_space}")
                    
                # Create a custom wrapper to handle the action/observation conversion
                return MultiAgentToSingleAgentWrapper(e, agent_id=i)
            
            # Create a vectorized environment for this agent
            from stable_baselines3.common.env_util import make_vec_env
            vec_env = make_vec_env(
                make_env,  # Use our environment factory
                n_envs=self.n_envs,  # Use multiple environments for faster training
                vec_env_cls=DummyVecEnv,  # Use DummyVecEnv for simplicity
            )
            self.vec_envs.append(vec_env)
            
            # Create PPO agent for this agent
            from stable_baselines3 import PPO
            ppo_agent = PPO(
                "MultiInputPolicy",
                vec_env,
                verbose=self.config.verbose,
                batch_size=self.config.batch_size,
                n_steps=self.config.num_steps,
                gamma=self.config.gamma,
                learning_rate=self.config.learning_rate,
                ent_coef=self.config.entropy_coef,
                clip_range=self.config.clip_param,
                n_epochs=self.config.ppo_epochs,
                vf_coef=self.config.value_loss_coef,
                max_grad_norm=self.config.max_grad_norm,
                gae_lambda=self.config.gae_lambda,
                device=self.device,
                # Additional parameters for better exploration
                target_kl=self.config.target_kl,  # Target KL divergence to prevent too large policy updates
                normalize_advantage=True  # Normalize advantages for more stable training
            )
            self.ppo_agents.append(ppo_agent)
        
        # For backward compatibility
        self.agents = self.ppo_agents

    def get_agent_obs(self, state, agent_id):
        """Extract observation for a specific agent"""
        # Check if state is already a dict (suitable for a single agent)
        if isinstance(state, dict):
            return state
        
        # Handle tuple/list observations from multi-agent env
        if isinstance(state, (tuple, list)) and len(state) > agent_id:
            agent_state = state[agent_id]
        
            # Convert to dict format if needed
            if not isinstance(agent_state, dict):
                return {
                    'image': agent_state,
                    'direction': np.array(0)
                }
            return agent_state
        
        # If state is a numpy array, assume it's an image
        if isinstance(state, np.ndarray):
            return {
                'image': state,
                'direction': np.array(0)
            }
        
        # Fallback for unexpected formats
        print(f"Warning: Unexpected state format in get_agent_obs: {type(state)}")
        # Return empty observation as a last resort
        return {
            'image': np.zeros((5, 5, 3), dtype=np.uint8),
            'direction': np.array(0)
        }

    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        """Run one episode"""
        viz_data = None
        if visualize:
            viz_data = self._init_visualization_data(env)
            print(f"[MetaController] Visualization enabled for episode {episode}")
            print(f"[MetaController] Visualization will use deterministic={visualize}")
            
        # Add debug prints for model loading status
        print(f"[MetaController] Agent models loaded status:")
        for i, agent in enumerate(self.agents):
            print(f"  Agent {i} loaded: {hasattr(agent, 'policy') and agent.policy is not None}")
        
        # Reset environment
        state = env.reset()
        
        # Print initial observation format
        print(f"[MetaController] Initial state type: {type(state)}")
        if isinstance(state, dict):
            print(f"  State keys: {state.keys()}")
        elif isinstance(state, (list, tuple)):
            print(f"  State is a list/tuple with {len(state)} elements")
            if len(state) > 0:
                print(f"  First element type: {type(state[0])}")
        
        # Initialize episode data
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        rewards = []  # For tracking reward over time
        steps = 0
        total_episode_reward = 0
        episode_start_time = time.time()
        
        # Use save_interval from ppo.yaml if available, or fall back to a default
        save_interval = getattr(self.config, 'save_interval', 10000)
        
        # For debugging: force non-deterministic during visualization
        force_explore = getattr(self.config, 'force_explore', False)
        if force_explore:
            print("[MetaController] FORCING EXPLORATION MODE (non-deterministic actions)")
            
        # Anti-stuck mechanism
        MAX_STUCK_STEPS = 5
        stuck_counter = 0
        last_actions = None
        
        # Increase episode length for both training and visualization
        MAX_STEPS = 500 if visualize else 1000  # Longer for training, bit shorter for visualization
        
        # Track repetitive actions for penalty
        action_history = [[] for _ in range(self.n_agents)]
        
        while not done and steps < MAX_STEPS:
            # Collect all agent actions
            actions = []
            for i in range(self.n_agents):
                agent = self.agents[i]
                agent_obs = self.get_agent_obs(state, i)
                
                # Debug print for observation format
                if steps == 0:
                    print(f"[DEBUG] Agent {i} observation shape/format:")
                    if isinstance(agent_obs, dict):
                        for k, v in agent_obs.items():
                            if isinstance(v, np.ndarray):
                                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                            elif isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                            else:
                                print(f"  {k}: {type(v)}")
                
                # For both visualization and training, we'll use the predict method
                # deterministic=False for training to include exploration
                
                # Anti-stuck mechanism: if repeating the same actions for too long, force exploration
                if stuck_counter >= MAX_STUCK_STEPS:
                    deterministic = False
                    if stuck_counter == MAX_STUCK_STEPS:
                        print(f"[DEBUG] Detected stuck pattern at step {steps}. Forcing exploration!")
                else:
                    # Use deterministic for visualization, non-deterministic for training
                    deterministic = visualize 
                
                with torch.no_grad():
                    action, _ = agent.predict(agent_obs, deterministic=deterministic)
                
                actions.append(action)
                
                # Store action in history for this agent (for repetitive action penalty)
                if train:
                    action_val = action.item() if hasattr(action, 'item') else int(action)
                    action_history[i].append(action_val)
                    # Keep only last 10 actions
                    if len(action_history[i]) > 10:
                        action_history[i].pop(0)
            
            # Check if actions are the same as last step (for stuck detection)
            current_actions_list = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
            if last_actions == current_actions_list:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_actions = current_actions_list
            
            # Debug print for actions
            if steps % 10 == 0 or steps < 5 or stuck_counter > 0:
                print(f"[DEBUG] Step {steps} actions: {current_actions_list} {'(Stuck!)' if stuck_counter > 0 else ''}")
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(actions)
            
            # Apply repetitive action penalty during training to encourage exploration
            if train:
                modified_reward = []
                for i, r in enumerate(reward if isinstance(reward, (list, tuple)) else [reward] * self.n_agents):
                    # Check for repetitive actions
                    if len(action_history[i]) >= 5:
                        last_5_actions = action_history[i][-5:]
                        # Count number of movement actions (0=left, 1=right, 2=forward)
                        movement_count = sum(1 for a in last_5_actions if a in [0, 1, 2])
                        
                        # If too few movement actions, apply penalty
                        if movement_count < 2:  # Less than 2 movement actions in last 5
                            r -= 0.01  # Small penalty for not moving
                            
                        # Check for same action repeated
                        if len(set(last_5_actions)) == 1:  # Same action repeated 5 times
                            r -= 0.02  # Larger penalty for repeating exact same action
                    
                    modified_reward.append(r)
                
                # Replace reward with modified version for training
                reward = modified_reward if len(modified_reward) > 1 else modified_reward[0]
            
            # Calculate total reward for this step
            step_reward = sum(reward) if isinstance(reward, (list, tuple)) else reward
            total_episode_reward += step_reward
            
            # For visualization purposes
            if visualize:
                if steps % 10 == 0:  # More frequent logs for debugging
                    print(f"[MetaController] Visualization step {steps}: Actions={current_actions_list}, Reward={step_reward:.2f}")
                viz_data = self._add_visualization_data(viz_data, env, state, actions, next_state)
                rewards.append(reward)
            
            # Store episode data
            episode_states.append(state)
            episode_actions.append(actions)
            episode_rewards.append(reward)
            
            # Update state
            state = next_state
            steps += 1
            self.total_steps += 1
            
            # Maybe update policy, save model, etc.
            if train and (done or self.total_steps % self.config.num_steps == 0):
                self._update_models()
            
            # Use save_interval instead of save_model_every
            if save_model and self.total_steps % save_interval == 0:
                print(f"[MetaController] Saving models at step {self.total_steps}")
                self._save_models(episode)
            
            if done:
                episode_duration = time.time() - episode_start_time
                print(f"[MetaController] Episode {episode} completed - {steps} steps, reward: {total_episode_reward:.2f}")
                # Print final actions summary
                print(f"[DEBUG] Final episode actions distribution:")
                action_counts = {}
                for step_actions in episode_actions:
                    for i, a in enumerate(step_actions):
                        a_val = a.item() if hasattr(a, 'item') else int(a)
                        key = f"agent_{i}_action_{a_val}"
                        action_counts[key] = action_counts.get(key, 0) + 1
                for k, v in action_counts.items():
                    print(f"  {k}: {v} times ({v/steps*100:.1f}%)")
                
                if log:
                    self._log_episode(episode, steps, episode_rewards)
                break
        
        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

    def _update_models(self):
        """Update all agent models - optimized for speed"""
        if self.total_steps >= self.config.initial_memory:
            if self.total_steps % self.config.update_every == 0:
                for agent_id, agent in enumerate(self.agents):
                    # Pass a larger num_timesteps value for faster learning
                    agent.learn(total_timesteps=self.config.num_steps * 2)
                if self.total_steps % 5000 == 0:  # Less frequent logging
                    print(f"[MetaController] Updated agent models at step {self.total_steps}")

    def _log_episode(self, episode, steps, rewards):
        """Log episode metrics to wandb"""
        rewards = np.array(rewards)
        total_reward = np.sum(rewards)
        wandb.log({
            "episode/x_axis": episode,
            "episode/reward": total_reward,
            "episode/length": steps,
        })

    def _save_models(self, episode):
        """Save model checkpoints"""
        save_model_episode = getattr(self.config, 'save_model_episode', 10000)
        if episode % save_model_episode == 0:
            print(f"[MetaController] Saving model checkpoints at episode {episode}")
            for i, agent in enumerate(self.agents):
                save_path = f"models/agent_{i}_episode_{episode}"
                agent.save(save_path)
            print(f"[MetaController] Checkpoint saving complete")

    def load_models(self, model_path=None):
        """Load saved models"""
        if not model_path:
            return
        print(f"[MetaController] Loading models from {model_path}")
        for i, agent in enumerate(self.agents):
            path = f"{model_path}_agent_{i}" if model_path else f"models/agent_{i}"
            try:
                agent.load(path)
                print(f"[MetaController] Loaded agent {i} from {path}")
            except Exception as e:
                print(f"[MetaController] Error loading agent {i}: {e}")

    def train(self, env):
        """Train all agents in a truly multi-agent fashion"""
        # Get n_episodes from config - prioritize the one from YAML
        if hasattr(self.config, 'n_episodes'):
            n_episodes = self.config.n_episodes
            print(f"[MetaController] Using n_episodes={n_episodes} from config")
        else:
            n_episodes = 1000  # Default fallback
            print(f"[MetaController] Warning: n_episodes not found in config, using default: {n_episodes}")
            
        print(f"\n[MetaController] Starting multi-agent training for {n_episodes} episodes")
        print(f"[MetaController] Config: batch_size={self.config.batch_size}, lr={self.config.learning_rate}, n_envs={self.n_envs}")
        print(f"[MetaController] Models will save every {getattr(self.config, 'save_model_episode', 10000)} episodes and every {getattr(self.config, 'save_interval', 10000)} steps")
        
        # Set up progress bar for episodes
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=n_episodes, desc="Training Progress", unit="episode")
        except ImportError:
            print("[MetaController] tqdm not installed. Install with 'pip install tqdm' for progress bar.")
            progress_bar = None
            
        # Create directories if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            "episodes": [],
            "collective_return": [],
        }
        
        for i in range(self.n_agents):
            self.metrics[f"agent_{i}_reward"] = []
        
        # Create learning curve figure
        self._init_learning_curves()
        
        # Set up the environment
        start_time = time.time()
        episode_reward = np.zeros(self.n_agents)
        episode_rewards_history = []
        episode_lengths = []
        
        try:
            # Main training loop - truly multi-agent
            for episode in range(1, n_episodes + 1):
                # Reset environment
                state = env.reset()
                episode_reward = np.zeros(self.n_agents)
                episode_step = 0
                done = False
                episode_start_time = time.time()
                
                # Initialize tracking for action repetition
                last_actions = [0] * self.n_agents
                
                # Run one episode
                while not done and episode_step < 500:  # Cap at 500 steps
                    # Get all agents' actions
                    actions = []
                    for i in range(self.n_agents):
                        # Get observation for this agent
                        agent_obs = self.get_agent_obs(state, i)
                        
                        # Get action from this agent's policy
                        action, _ = self.ppo_agents[i].predict(agent_obs, deterministic=False)
                        
                        # Ensure action is valid
                        if isinstance(action, np.ndarray):
                            action = action.item()
                        action = int(action) % 7
                        
                        actions.append(action)
                    
                    # Step environment with all actions together
                    next_state, rewards, done, info = env.step(actions)
                    
                    # Apply reward shaping to encourage movement and exploration
                    shaped_rewards = []
                    for i, r in enumerate(rewards if isinstance(rewards, (list, tuple)) else [rewards] * self.n_agents):
                        # Original reward
                        shaped_r = r
                        
                        # Bonus for movement actions
                        if actions[i] in [0, 1, 2]:  # left, right, forward
                            shaped_r += 0.01  # Small bonus for movement
                            
                        # Penalty for typical "stuck" actions
                        if actions[i] in [3, 4, 5]:  # pickup, drop, toggle
                            # Count how many times these actions have been taken
                            if episode_step > 0 and i < len(last_actions) and last_actions[i] == actions[i]:
                                shaped_r -= 0.005  # Small penalty for repeating non-movement actions
                                
                        shaped_rewards.append(shaped_r)
                    
                    # Store last actions for comparison in next step
                    last_actions = actions.copy()
                    
                    # Update episode rewards with shaped rewards
                    episode_reward += np.array(shaped_rewards) if len(shaped_rewards) > 1 else np.array([shaped_rewards[0]] * self.n_agents)
                    
                    # Update state
                    state = next_state
                    episode_step += 1
                    
                    # Every 100 steps, update the learning curves
                    if episode_step % 100 == 0:
                        self._update_learning_curves()
                
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"Total Reward": f"{np.sum(episode_reward):.2f}"})
                
                # Log episode results
                episode_rewards_history.append(episode_reward)
                episode_lengths.append(episode_step)
                episode_duration = time.time() - episode_start_time
                
                # Update metrics
                self.metrics["episodes"].append(episode)
                self.metrics["collective_return"].append(np.sum(episode_reward))
                for i in range(self.n_agents):
                    self.metrics[f"agent_{i}_reward"].append(episode_reward[i])
                
                # Log episode summary for key milestones
                if episode % max(1, min(100, n_episodes // 10)) == 0:
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"[MetaController] Episode {episode}/{n_episodes} - Duration: {episode_duration:.2f}s, Steps: {episode_step}, Total reward: {np.sum(episode_reward):.2f}")
                    print(f"[MetaController] Total training time so far: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                
                # Log to wandb every 10 episodes
                if episode % 10 == 0 and wandb.run is not None:
                    log_dict = {
                        "episode": episode,
                        "mean_length": np.mean(episode_lengths[-100:]),
                        "collective_return": np.sum(episode_reward),
                        "progress": episode / n_episodes,
                    }
                    
                    for i in range(self.n_agents):
                        log_dict[f"agent_{i}/reward"] = episode_reward[i]
                        log_dict[f"agent_{i}/mean_reward"] = np.mean([r[i] for r in episode_rewards_history[-100:]])
                    
                    wandb.log(log_dict)
                
                # Update learning curves and print progress every 100 episodes
                if episode % 100 == 0:
                    self._update_learning_curves()
                    mean_100ep_reward = [np.mean([r[i] for r in episode_rewards_history[-100:]]) for i in range(self.n_agents)]
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    
                    print(f"[MetaController] Stats after {episode} episodes: "
                          f"Mean 100ep rewards: {[f'{r:.2f}' for r in mean_100ep_reward]}, "
                          f"Collective: {np.sum(mean_100ep_reward):.2f}")
                    print(f"[MetaController] Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s ({elapsed_time:.2f}s total)")
                
                # Train all agents using their collected experiences
                if episode % 10 == 0:  # Train every 10 episodes
                    for i in range(self.n_agents):
                        # Create mini-batch of experiences for this agent
                        self.ppo_agents[i].learn(
                            total_timesteps=1000,  # Small number of steps 
                            reset_num_timesteps=False,  # Don't reset timestep counter
                            progress_bar=False
                        )
                
                # Save models periodically
                save_model_episode = getattr(self.config, 'save_model_episode', 10000)
                if episode % save_model_episode == 0:
                    print(f"[MetaController] Saving model checkpoint at episode {episode}...")
                    for i in range(self.n_agents):
                        save_path = f"models/agent_{i}_episode_{episode}"
                        self.ppo_agents[i].save(save_path)
                
                # Save final models
                if episode % 1000 == 0:
                    print(f"[MetaController] Saving regular checkpoint at episode {episode}")
                    for i in range(self.n_agents):
                        save_path = f"models/agent_{i}_ppo"
                        self.ppo_agents[i].save(save_path)
                        
                # Check if we've reached the target number of episodes
                if episode >= n_episodes:
                    print(f"[MetaController] Reached target of {n_episodes} episodes. Training complete.")
                    break
            
            # Final save when training completes
            print(f"[MetaController] Saving final models after {n_episodes} episodes")
            for i in range(self.n_agents):
                save_path = f"models/agent_{i}_final"
                self.ppo_agents[i].save(save_path)
                
        except KeyboardInterrupt:
            print(f"[MetaController] Training interrupted at episode {episode}. Saving current models...")
            for i in range(self.n_agents):
                self.ppo_agents[i].save(f"models/agent_{i}_interrupted")
        finally:
            # Close progress bar
            if progress_bar is not None:
                progress_bar.close()
                
            # Training complete, generate final learning curves
            self._create_learning_curves(final=True)
            training_time = time.time() - start_time
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"[MetaController] Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Close environments to release resources
            for vec_env in self.vec_envs:
                vec_env.close()
            
            # Return final models to signal completion
            return self.ppo_agents

    def _init_learning_curves(self):
        """Initialize learning curves"""
        try:
            import matplotlib.pyplot as plt
            
            self.fig, self.ax = plt.figure(figsize=(10, 6)), plt.gca()
            self.ax.set_xlabel("Training Steps")
            self.ax.set_ylabel("Average Reward")
            self.ax.set_title("Agent Learning Curves")
            
            # Initialize empty lines
            self.plot_lines = {}
            for i in range(self.n_agents):
                line, = self.ax.plot([], [], label=f"Agent {i}")
                self.plot_lines[f"agent_{i}"] = line
            
            self.ax.legend()
            plt.savefig("plots/learning_curves_initial.png")
            
            if wandb.run is not None:
                wandb.log({"learning_curves": wandb.Image("plots/learning_curves_initial.png")})
            
        except Exception as e:
            print(f"Error setting up learning curves: {e}")

    def _update_learning_curves(self):
        """Update learning curves during training"""
        try:
            import matplotlib.pyplot as plt
            
            # Update each line
            for i in range(self.n_agents):
                key = f"agent_{i}_reward"
                if key in self.metrics and len(self.metrics[key]) > 0:
                    x_data = self.metrics["episodes"]
                    y_data = self.metrics[key]
                    
                    if len(x_data) != len(y_data):
                        # Ensure equal lengths
                        min_len = min(len(x_data), len(y_data))
                        x_data = x_data[:min_len]
                        y_data = y_data[:min_len]
                    
                    self.plot_lines[f"agent_{i}"].set_data(x_data, y_data)
            
            # Update plot limits
            if len(self.metrics["episodes"]) > 0:
                self.ax.set_xlim(0, max(self.metrics["episodes"]) * 1.1)
                
                # Find reasonable y limits
                all_rewards = []
                for i in range(self.n_agents):
                    key = f"agent_{i}_reward"
                    if key in self.metrics:
                        all_rewards.extend(self.metrics[key])
                
                if all_rewards:
                    self.ax.set_ylim(min(all_rewards) * 1.1, max(all_rewards) * 1.1)
            
            # Save updated plot
            self.fig.canvas.draw()
            plt.savefig("plots/learning_curves_current.png")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({"learning_curves": wandb.Image("plots/learning_curves_current.png")})
            
        except Exception as e:
            print(f"Error updating learning curves: {e}")

    def _create_learning_curves(self, final=False):
        """Generate final learning curves"""
        try:
            import matplotlib.pyplot as plt
            
            # Create new figure for final plot
            plt.figure(figsize=(12, 8))
            
            # Plot each agent's learning curve
            for i in range(self.n_agents):
                key = f"agent_{i}_reward"
                if key in self.metrics and len(self.metrics[key]) > 0:
                    x_data = self.metrics["episodes"]
                    y_data = self.metrics[key]
                    
                    if len(x_data) != len(y_data):
                        # Ensure equal lengths
                        min_len = min(len(x_data), len(y_data))
                        x_data = x_data[:min_len]
                        y_data = y_data[:min_len]
                    
                    plt.plot(x_data, y_data, marker='.', linestyle='-', label=f"Agent {i}")
            
            plt.xlabel("Training Steps")
            plt.ylabel("Average Reward")
            plt.title("MultiGrid Agent Learning Curves")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save final high-quality plot
            filename = "plots/learning_curves_final.png" if final else "plots/learning_curves.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "final_learning_curves" if final else "learning_curves": 
                    wandb.Image(filename)
                })
            
        except Exception as e:
            print(f"Error creating learning curves: {e}")

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        """Create visualization video"""
        print(f"[MetaController] Starting visualization in {video_dir}")
        print(f"[MetaController] Model info before visualization:")
        for i, agent in enumerate(self.agents):
            print(f"  Agent {i} model type: {type(agent)}")
            if hasattr(agent, 'policy'):
                print(f"  Agent {i} policy type: {type(agent.policy)}")
            
        if not viz_data:
            print("[MetaController] Running episode for visualization with deterministic=True")
            viz_data = self.run_one_episode(env, episode=0, log=False, train=False, 
                                          save_model=False, visualize=True)
        
        # Set up video directory
        video_path = os.path.join(video_dir, self.config.domain)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        print(f"[MetaController] Video will be saved to {video_path}")
        
        # Get action names
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name
        print(f"[MetaController] Action mapping: {action_dict}")
        
        # Create frames
        traj_len = len(viz_data['rewards'])
        print(f"[MetaController] Creating {traj_len} visualization frames")
        # Print some statistics about actions
        if 'actions' in viz_data and len(viz_data['actions']) > 0:
            all_actions = []
            # Convert numpy arrays to integers before adding to list
            for step_actions in viz_data['actions']:
                step_actions_int = []
                for a in step_actions:
                    # Convert numpy array to integer if needed
                    if hasattr(a, 'item'):
                        step_actions_int.append(a.item())
                    else:
                        step_actions_int.append(int(a))
                all_actions.extend(step_actions_int)
            
            action_freq = {}
            for a in all_actions:
                action_freq[a] = action_freq.get(a, 0) + 1
            print(f"[MetaController] Action frequency in trajectory: {action_freq}")
        
        for t in range(traj_len):
            self._visualize_frame(t, viz_data, action_dict, video_path)
            if t % 25 == 0:
                print(f"[MetaController] Creating frame {t}/{traj_len}")
                if t > 0:
                    # Convert numpy arrays to integers for printing
                    actions_to_print = []
                    for a in viz_data['actions'][t-1]:
                        if hasattr(a, 'item'):
                            actions_to_print.append(a.item())
                        else:
                            actions_to_print.append(int(a))
                    print(f"  Actions at this step: {actions_to_print}")
                    print(f"  Rewards at this step: {viz_data['rewards'][t-1]}")
        
        # Create video
        video_name = f'{mode}_trajectory_video'
        print(f"[MetaController] Creating final video: {video_name}")
        make_video(video_path, video_name)
        print(f"[MetaController] Visualization complete: {os.path.join(video_path, video_name)}.mp4")

    def _init_visualization_data(self, env):
        """Initialize data structure for visualization"""
        return {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [env.render('rgb_array')],
        }

    def _add_visualization_data(self, viz_data, env, state, actions, next_state):
        """Add frame data to visualization"""
        # Convert numpy arrays to integers before storing
        actions_int = []
        for a in actions:
            if hasattr(a, 'item'):
                actions_int.append(a.item())
            else:
                actions_int.append(int(a))
        viz_data['actions'].append(actions_int)
        
        # Process agent observations for visualization
        agent_images = []
        for i in range(self.n_agents):
            agent_obs = self.get_agent_obs(state, i)
            agent_image = agent_obs['image']
            
            # Convert list to numpy array if needed
            if isinstance(agent_image, list):
                agent_image = np.array(agent_image)
                
            # Ensure agent_image is a numpy array, not a tensor
            if isinstance(agent_image, torch.Tensor):
                agent_image = agent_image.cpu().numpy()
            
            # IMPORTANT: Extract this agent's observation if observations are stacked for all agents
            if len(agent_image.shape) == 4 and agent_image.shape[0] == self.n_agents:  
                # If first dimension matches number of agents, extract this agent's observation
                agent_image = agent_image[i]  # Select the observation for this agent
                print(f"Using agent {i}'s observation from stacked array: shape={agent_image.shape}")
            
            # Only attempt rendering if we have a valid numpy array
            try:
                rendered_image = env.get_obs_render(agent_image)
                agent_images.append(rendered_image)
            except Exception as e:
                print(f"ERROR: Could not render agent {i} observation: {e}")
                print(f"Agent {i} observation shape after processing: {agent_image.shape}")
                # Use a blank image as fallback
                agent_images.append(np.zeros((84, 84, 3), dtype=np.uint8))
        
        viz_data['agents_partial_images'].append(agent_images)
        
        # Capture full environment image
        try:
            viz_data['full_images'].append(env.render('rgb_array'))
        except Exception as e:
            print(f"Warning: Could not render full environment: {e}")
            # Use last image or create blank image as fallback
            if len(viz_data['full_images']) > 0:
                viz_data['full_images'].append(viz_data['full_images'][-1])
            else:
                viz_data['full_images'].append(np.zeros((500, 500, 3), dtype=np.uint8))
        
        return viz_data

    def _visualize_frame(self, t, viz_data, action_dict, video_path):
        """Create visualization for a single frame"""
        plot_single_frame(
            t, 
            viz_data['full_images'][t], 
            viz_data['agents_partial_images'][t], 
            viz_data['actions'][t], 
            viz_data['rewards'], 
            action_dict, 
            video_path, 
            self.config.model_name,
            predicted_actions=None
        )

