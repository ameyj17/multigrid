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

from utils import plot_single_frame, make_video, extract_mode_from_path
from networks.multigrid_network import MultiGridNetwork

class DictFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor that passes dict observations directly to the policy network"""
    
    def __init__(self, observation_space, features_dim=1):
        # Features dim is arbitrary here since we're not using it
        super().__init__(observation_space, features_dim)
        
    def forward(self, observations):
        """Simply pass the observations dictionary through"""
        return observations

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
        if isinstance(obs, dict):
            # Process each component of the observation
            processed_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    processed_obs[key] = torch.FloatTensor(value).to(self.device)
                else:
                    processed_obs[key] = value
            return processed_obs
        return obs
        
    def _policy_forward_dict(self, obs_dict):
        """Custom forward pass for the policy to handle dictionary observations"""
        return self.actor_net(obs_dict)
    
    def forward(self, obs):
        """
        Forward pass in all the networks (actor and critic)
        
        :param obs: Observation
        :return: action, value and log probability of the action
        """
        # Run the observation through the actor network to get action logits
        actor_features = self.actor_net(obs)
        
        # Create categorical distribution directly with logits
        # Note: Categorical distribution only needs logits, not mean_actions and log_std
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
            # print("[multiagent_metacontroller.predict] actions: " + str(actions))
        
        return actions, None

class SingleAgentEnv(gym.Wrapper):
    """Wrapper that adapts a multi-agent environment for a single agent"""
    def __init__(self, env, agent_id, n_agents, decentralized=True):
        super().__init__(env)
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.decentralized = decentralized
        self.last_actions = np.zeros(n_agents, dtype=np.int8)
        
        # Define fixed dimensions for ALL observation components
        self.image_shape = (5, 5, 3)  # Fixed shape for consistency
        self.direction_shape = (1,)   # Direction as a 1D array with 1 element
        self.other_actions_shape = (n_agents-1,)  # One slot per other agent
        
        # Define action space
        self.action_space = spaces.Discrete(7)  # Actions 0-6 are valid
        
        # Build observation space with shapes that EXACTLY match what we'll return
        base_obs_space = {
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=self.image_shape, 
                dtype=np.uint8
            ),
            'direction': spaces.Box(
                low=0,
                high=3,
                shape=self.direction_shape,  # Must be (1,)
                dtype=np.int64
            )
        }
        
        if not self.decentralized:
            # Add other agents' actions to observation space
            base_obs_space['other_actions'] = spaces.Box(
                low=0, 
                high=6, 
                shape=self.other_actions_shape,
                dtype=np.int8
            )
            
        self.observation_space = spaces.Dict(base_obs_space)
        
        # Log the exact observation space for debugging
        print(f"Agent {agent_id} observation space: {self.observation_space}")
    
    def _ensure_image_shape(self, img):
        """Ensure image has correct shape and type"""
        if img is None:
            return np.zeros(self.image_shape, dtype=np.uint8)
            
        # Convert to numpy if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img, dtype=np.uint8)
            
        # Handle batched inputs - we need to extract just one image
        if len(img.shape) == 4:
            img = img[0]  # Take the first image from the batch
            
        # Fix dimensions if needed
        if img.shape != self.image_shape:
            # We explicitly need output in HWC format (height, width, channels)
            # If channels are first, transpose them to last
            if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))  # Convert to HWC
                
            # Other dimension handling
            if len(img.shape) == 2:  # 2D grayscale
                img = np.stack([img] * 3, axis=-1)  # Convert to RGB
            elif len(img.shape) == 3 and img.shape[2] != 3:
                if img.shape[2] > 3:
                    img = img[:, :, :3]  # Take first 3 channels
                else:  # Not enough channels
                    temp = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
                    temp[:, :, :img.shape[2]] = img
                    img = temp
                
            # Resize if needed
            if img.shape[0] != self.image_shape[0] or img.shape[1] != self.image_shape[1]:
                resized = np.zeros(self.image_shape, dtype=np.uint8)
                h = min(img.shape[0], self.image_shape[0])
                w = min(img.shape[1], self.image_shape[1])
                resized[:h, :w, :] = img[:h, :w, :]
                img = resized
                
        # Ensure output is in HWC format with shape (5,5,3)
        return img.astype(np.uint8)
    
    def _ensure_direction_shape(self, direction):
        """Ensure direction has correct shape (1,) and type"""
        # Convert to int first
        if isinstance(direction, (list, tuple)):
            if len(direction) > 0:
                direction = int(direction[0])
            else:
                direction = 0
        elif isinstance(direction, np.ndarray):
            if direction.size > 0:
                direction = int(direction.item(0))
            else:
                direction = 0
        else:
            direction = int(direction) if direction is not None else 0
            
        # Return as numpy array with shape (1,)
        return np.array([direction], dtype=np.int64)
    
    def _ensure_other_actions_shape(self, actions):
        """Ensure other_actions has correct shape (n_agents-1,) and type"""
        if actions is None:
            return np.zeros(self.other_actions_shape, dtype=np.int8)
            
        if isinstance(actions, (list, tuple)):
            actions = np.array(actions, dtype=np.int8)
            
        # Ensure correct length
        if actions.shape != self.other_actions_shape:
            correct_actions = np.zeros(self.other_actions_shape, dtype=np.int8)
            # Copy as many elements as possible
            copy_len = min(len(actions), self.n_agents-1)
            correct_actions[:copy_len] = actions[:copy_len]
            actions = correct_actions
            
        return actions.astype(np.int8)
    
    def reset(self, **kwargs):
        """Reset the environment and return observation for this agent"""
        # Handle seed if provided
        if 'seed' in kwargs:
            obs = self.env.reset(seed=kwargs.get('seed'))
        else:
            obs = self.env.reset()
            
        self.last_actions = np.zeros(self.n_agents, dtype=np.int8)
        
        # Extract raw observation
        if isinstance(obs, (tuple, list)) and len(obs) > self.agent_id:
            agent_raw_obs = obs[self.agent_id]
            if isinstance(agent_raw_obs, dict) and 'image' in agent_raw_obs:
                image = agent_raw_obs['image']
                direction = agent_raw_obs.get('direction', 0)
            else:
                image = agent_raw_obs
                direction = 0
        elif isinstance(obs, dict) and 'image' in obs:
            image = obs['image']
            direction = obs.get('direction', 0)
        else:
            image = np.zeros(self.image_shape, dtype=np.uint8)
            direction = 0
        
        # Process components with shape enforcement
        image = self._ensure_image_shape(image)
        direction = self._ensure_direction_shape(direction)
        
        # Build consistent observation
        agent_obs = {
            'image': image,
            'direction': direction
        }
        
        # Add other agents' actions in coordinated mode
        if not self.decentralized:
            agent_obs['other_actions'] = np.zeros(self.other_actions_shape, dtype=np.int8)
            
        return agent_obs
    
    def step(self, action):
        """Take a step in the environment for this agent"""
        # Ensure action is a valid integer
        action_int = int(action) if np.isscalar(action) else int(action[0])
        
        # Update this agent's action in the joint action
        full_action = self.last_actions.copy()
        full_action[self.agent_id] = action_int
        
        # Save for next observation
        self.last_actions = full_action.copy()
        
        # Step environment with joint action
        next_obs, rewards, done, info = self.env.step(full_action.tolist())
        
        # Extract raw observation for this agent
        if isinstance(next_obs, (tuple, list)) and len(next_obs) > self.agent_id:
            agent_raw_obs = next_obs[self.agent_id]
            if isinstance(agent_raw_obs, dict) and 'image' in agent_raw_obs:
                image = agent_raw_obs['image']
                direction = agent_raw_obs.get('direction', 0)
            else:
                image = agent_raw_obs
                direction = 0
        elif isinstance(next_obs, dict) and 'image' in next_obs:
            image = next_obs['image']
            direction = next_obs.get('direction', 0)
        else:
            image = np.zeros(self.image_shape, dtype=np.uint8)
            direction = 0
            
        # Process components with shape enforcement
        image = self._ensure_image_shape(image)
        direction = self._ensure_direction_shape(direction)
        
        # Build consistent observation
        agent_obs = {
            'image': image,
            'direction': direction
        }
        
        # Add other agents' actions in coordinated mode
        if not self.decentralized:
            other_actions = []
            for i in range(self.n_agents):
                if i != self.agent_id:
                    other_actions.append(self.last_actions[i])
            agent_obs['other_actions'] = self._ensure_other_actions_shape(other_actions)
        
        # Extract reward for this agent
        if isinstance(rewards, (tuple, list)) and len(rewards) > self.agent_id:
            agent_reward = rewards[self.agent_id]
        else:
            agent_reward = rewards
            
        return agent_obs, agent_reward, done, info

class MetaController:
    """Coordinates multiple PPO agents with the multigrid environment"""
    def __init__(self, config, env, device, training=True, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.debug = debug
        self.n_agents = env.n_agents if hasattr(env, 'n_agents') else 3
        self.total_steps = 0
        
        # Create single agent environments
        self.agent_envs = []
        self.agents = []
        
        # Determine training mode
        self.decentralized = getattr(config, 'decentralized_training', True)
        print(f"Training mode: {'Decentralized' if self.decentralized else 'Coordinated'}")
        
        for i in range(self.n_agents):
            # Create wrapped environment for each agent with training mode
            agent_env = SingleAgentEnv(env, i, self.n_agents, decentralized=self.decentralized)
            self.agent_envs.append(agent_env)
            
            # Ensure the environment is properly vectorized with DummyVecEnv
            vec_env = DummyVecEnv([lambda i=i: self.agent_envs[i]])
            
            # Create PPO agent with explicitly set dimensions
            agent = PPO(
                policy=CustomPolicy,
                env=vec_env,
                learning_rate=config.learning_rate,
                n_steps=config.num_steps,
                batch_size=config.batch_size,
                n_epochs=config.ppo_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_param,
                ent_coef=config.entropy_coef,
                vf_coef=config.value_loss_coef,
                max_grad_norm=config.max_grad_norm,
                policy_kwargs={
                    "config": config,
                    "n_agents": self.n_agents,
                    "agent_id": i
                },
                device=device
            )
            
            # Explicitly set action dimensions on the rollout buffer
            agent.rollout_buffer.action_dim = 1
            
            self.agents.append(agent)

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

    def run_one_episode(self, env, episode=0, log=True, train=True, 
                        save_model=True, visualize=False):
        """Run a single episode, collecting experiences and optionally training"""
        state = env.reset()
        done = False
        rewards = []
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_infos = []
        
        # Visualization data
        if visualize:
            viz_data = self._init_visualization_data(env)
        
        while not done:
            self.total_steps += 1
            
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self.get_agent_obs(state, i)
                # Always use deterministic=True for prediction
                action, _ = agent.predict(agent_obs, deterministic=True)
                
                # Convert action to appropriate format for environment
                if isinstance(action, np.ndarray):
                    # Clip to valid action range for discrete actions (0-6)
                    action = np.clip(action, 0, 6).astype(int)
                    
                    # If it's a vector action, take the first element
                    if action.size > 1:
                        action = action[0]
                    
                    # Convert to int for discrete action space
                    if isinstance(action, np.ndarray) and action.size == 1:
                        action = action.item()
                
                # Ensure action is an integer
                action = int(action)
                actions.append(action)
            
            next_state, reward, done, info = env.step(actions)
            
            # Store experiences for later PPO update (no direct replay buffer access needed)
            if train:
                episode_states.append(state)
                episode_actions.append(actions)
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_infos.append(info)
            
            # Update visualization data if needed
            if visualize:
                viz_data = self._add_visualization_data(viz_data, env, state, actions, next_state)
            
            state = next_state
            rewards.append(reward)
            
            # Update models if it's time
            if train and (done or self.total_steps % self.config.num_steps == 0):
                self._update_models()
        
        # Log episode results
        if log:
            self._log_episode(episode, len(rewards), rewards)
        
        # Save models if needed
        if save_model:
            self._save_models(episode)
        
        # Return visualization data if requested
        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

    def _update_models(self):
        """Update all agent models"""
        if self.total_steps >= self.config.initial_memory:
            if self.total_steps % self.config.update_every == 0:
                for agent in self.agents:
                    agent.learn(total_timesteps=self.config.num_steps)

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
        if episode % self.config.save_model_episode == 0:
            for i, agent in enumerate(self.agents):
                agent.save(f"models/agent_{i}_episode_{episode}")

    def load_models(self, model_path=None):
        """Load saved models"""
        if not model_path:
            return
        for i, agent in enumerate(self.agents):
            path = f"{model_path}_agent_{i}" if model_path else f"models/agent_{i}"
            agent.load(path)

    def train(self, env):
        """Main training loop"""
        # Override config.n_episodes if passed from command line
        if hasattr(self.config, 'n_episodes'):
            n_episodes = self.config.n_episodes
            print("[metacontroller.train] n_episodes: " + str(n_episodes))
        else:
            n_episodes = 100000  # Ensure 100,000 episodes
        
        print(f"Starting training for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            # Print progress periodically
            if episode % self.config.print_every == 0:
                print(f"Episode {episode}/{n_episodes}")
            
            # Visualize occasionally
            if episode % self.config.visualize_every == 0 and not self.debug:
                print(f"Creating visualization at episode {episode}")
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, f"{self.config.mode}_training_step{episode}", viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        """Create visualization video"""
        if not viz_data:
            viz_data = self.run_one_episode(env, episode=0, log=False, train=False, 
                                          save_model=False, visualize=True)
        
        # Set up video directory
        video_path = os.path.join(video_dir, self.config.experiment_name, self.config.model_name)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # Get action names
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name
        
        # Create frames
        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self._visualize_frame(t, viz_data, action_dict, video_path)
            print(f'Frame {t}/{traj_len}')
        
        # Create video
        make_video(video_path, f'{mode}_trajectory_video')

    def _init_visualization_data(self, env):
        """Initialize data structure for visualization"""
        return {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [env.render('rgb_array')],
        }

    def _add_visualization_data(self, viz_data, env, state, actions, next_state):
        """Add frame data to visualization"""
        viz_data['actions'].append(actions)
        
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
                
            # Only attempt rendering if we have a valid numpy array
            try:
                rendered_image = env.get_obs_render(agent_image)
                agent_images.append(rendered_image)
            except Exception as e:
                # print(f"Warning: Could not render agent {i} observation: {e}")
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

