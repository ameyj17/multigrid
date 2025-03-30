from copy import deepcopy
import gym
import gymnasium
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

# CustomMultiGridPolicy is intentionally not imported - see end of file for implementation notes
# from networks.multigrid_network import CustomMultiGridPolicy 

from utils import plot_single_frame, make_video, extract_mode_from_path

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import DictRolloutBuffer # Keep for commented line below
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the shimmy compatibility wrapper
import shimmy

from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces

# Add SingleAgentObsWrapper for custom_multigrid policy
# This wrapper is not used in the current simplified implementation
# See notes at end of file for CustomMultiGridPolicy implementation details
"""
class SingleAgentObsWrapper(gymnasium.ObservationWrapper):
    # Wrapper that modifies the observation space reported by a Dict environment
    # to represent a single agent's view, assuming the 'image' key holds
    # the multi-agent observation [N, H, W, C]. 
    # Not used in this simplified implementation.
"""

# Add a proper Gymnasium adapter that directly inherits from Env
class GymToGymnasiumAdapter(gymnasium.Env):
    """Direct adapter from Gym to Gymnasium."""
    
    def __init__(self, gym_env):
        super().__init__()
        self.gym_env = gym_env
        self.n_agents = getattr(gym_env, 'n_agents', 1)  # Pass through n_agents for multi-agent envs
        
        # Convert observation space
        if isinstance(gym_env.observation_space, gym_spaces.Dict):
            self.observation_space = gymnasium_spaces.Dict({
                k: self._convert_space(v) for k, v in gym_env.observation_space.spaces.items()
            })
        else:
            self.observation_space = self._convert_space(gym_env.observation_space)
        
        # Convert action space
        self.action_space = self._convert_space(gym_env.action_space)
        
        # Add any other attributes we need to access directly
        self._gym_env_attrs = {}
        for attr in ['get_obs_render', 'Actions', 'render', 'close']:
            if hasattr(gym_env, attr):
                self._gym_env_attrs[attr] = getattr(gym_env, attr)
    
    def _convert_space(self, space):
        """Convert gym space to gymnasium space."""
        if isinstance(space, gym_spaces.Box):
            return gymnasium_spaces.Box(low=space.low, high=space.high, 
                                      shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gym_spaces.Discrete):
            return gymnasium_spaces.Discrete(n=space.n)
        elif isinstance(space, gym_spaces.Tuple):
            return gymnasium_spaces.Tuple(
                tuple(self._convert_space(s) for s in space.spaces))
        elif isinstance(space, gym_spaces.Dict):
            return gymnasium_spaces.Dict(
                {k: self._convert_space(s) for k, s in space.spaces.items()})
        else:
            # For any other space type, try to pass through
            return space
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            # Try to set seed if available in gym env
            try:
                self.gym_env.seed(seed)
            except (AttributeError, TypeError):
                pass
        
        # Call gym env reset with appropriate kwargs
        if options is not None:
            obs = self.gym_env.reset(**options)
        else:
            obs = self.gym_env.reset()
        
        return obs, {}  # Add empty info dict for gymnasium API
    
    def step(self, action):
        """Step environment."""
        # Convert action if needed - For multi-agent envs, need special handling
        # VecEnv will likely pass a single integer or list with one element, 
        # but our original env expects a list with actions for all agents
        action_to_pass = action
        if getattr(self.gym_env, 'n_agents', 1) > 1:
            # If we get a single action (from vectorized env), duplicate it for all agents
            if not isinstance(action, (list, tuple, np.ndarray)) or len(action) == 1:
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 1:
                    action_value = action[0]  # Extract from list with one element
                else:
                    action_value = action  # Use as is
                action_to_pass = [action_value] * self.n_agents
        
        # Call gym env step
        try:
            if getattr(self, '_just_reset', False):
                obs = self.gym_env.reset()
                reward = 0.0
                done = False
                info = {}
                self._just_reset = False
            else:
                obs, reward, done, info = self.gym_env.step(action_to_pass)
        except Exception as e:
            print(f"Original env step error: {e}")
            # Provide fallback values
            obs = None
            reward = 0.0
            done = True
            info = {}
        
        # Convert to gymnasium API
        terminated = done
        truncated = info.get('TimeLimit.truncated', False)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment."""
        # Forward to gym env render
        render_fn = self._gym_env_attrs.get('render', None)
        if render_fn is not None:
            return render_fn(mode)
        return None
    
    def close(self):
        """Close environment."""
        if hasattr(self.gym_env, 'close'):
            return self.gym_env.close()
    
    def get_obs_render(self, obs):
        """Forward get_obs_render if available."""
        render_fn = self._gym_env_attrs.get('get_obs_render', None)
        if render_fn is not None:
            return render_fn(obs)
        return None
    
    def __getattr__(self, name):
        """Forward attribute access to gym_env for any other attributes."""
        if name in self._gym_env_attrs:
            return self._gym_env_attrs[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")


class GymToGymnasiumDictWrapper(gym.Wrapper):
    """Converts Gym Dict spaces/API to Gymnasium format."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym_spaces.Dict), "Requires Dict obs space."
        self.observation_space = gymnasium_spaces.Dict({
            key: self._convert_space(space) for key, space in env.observation_space.spaces.items()
        })
        self.action_space = self._convert_space(env.action_space)

    def _convert_space(self, space):
        if isinstance(space, gym_spaces.Box): return gymnasium_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gym_spaces.Discrete): return gymnasium_spaces.Discrete(n=space.n)
        elif isinstance(space, gym_spaces.Tuple): return gymnasium_spaces.Tuple(tuple(self._convert_space(s) for s in space.spaces))
        elif isinstance(space, gym_spaces.Dict): return gymnasium_spaces.Dict({key: self._convert_space(s) for key, s in space.spaces.items()})
        else: return space # Pass through if unknown

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs); return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done; truncated = info.get('TimeLimit.truncated', False)
        return obs, reward, terminated, truncated, info


class DiscreteActionWrapper(gymnasium.Wrapper):
    """Ensures actions are int and rewards are scalar."""
    def __init__(self, env):
        super().__init__(env)
        # Determine n_actions based on the original env's action space
        original_action_space = env.action_space
        if isinstance(original_action_space, (gym_spaces.Discrete, gymnasium_spaces.Discrete)):
            n_actions = original_action_space.n
        else:
            # You might need a more robust way to determine n_actions
            # if the original space isn't Discrete. For MultiGrid, 7 is common.
            print("Warning: Original action space is not Discrete. Assuming 7 actions for DiscreteActionWrapper.")
            n_actions = 7 # Default assumption for MultiGrid
        self.action_space = gymnasium_spaces.Discrete(n_actions)

    def step(self, action):
        # Action should already be compatible if PPO uses the Discrete space
        # Ensure it's an integer if coming from external source
        try:
            action_int = int(action)
        except (ValueError, TypeError):
            action_int = action # Pass through if already int or other type

        # Call the wrapped env's step
        obs, reward, terminated, truncated, info = self.env.step(action_int)

        # Ensure reward is scalar (assuming multi-agent reward might be a list/array)
        if isinstance(reward, (list, tuple, np.ndarray)) and len(reward) > 0:
             scalar_reward = float(reward[0]) # Take the first agent's reward if vector
        elif isinstance(reward, (int, float)):
             scalar_reward = float(reward)
        else:
             print(f"Warning: Unexpected reward type in DiscreteActionWrapper: {type(reward)}, value: {reward}. Using 0.0.")
             scalar_reward = 0.0

        # Gymnasium step return order
        return obs, scalar_reward, terminated, truncated, info

class MultiAgentPPOController():
    """Manages multiple PPO agents for a multi-agent environment."""
    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.debug = debug
        self.n_agents = env.n_agents
        self.default_model_path = os.path.join('models', config.experiment_name, config.model_name)
        print(f"Initializing MultiAgentPPOController ({self.n_agents} agents).")

        # --- Initialize Environment ---
        # Wrap the environment to make it compatible with gymnasium
        wrapped_env = DiscreteActionWrapper(GymToGymnasiumAdapter(self.env))
        
        # Debug info
        print(f"Environment observation space: {wrapped_env.observation_space}")
        print(f"Environment action space: {wrapped_env.action_space}")
        
        # --- Initialize PPO Agents ---
        self.agents = []
        for i in range(self.n_agents):
            try:
                print(f"Creating PPO agent {i} with MultiInputPolicy")
                agent = PPO(
                    policy="MultiInputPolicy",
                    env=wrapped_env,  # All agents share the same environment
                    verbose=0,
                    device=self.device,
                    gamma=config.get('gamma', 0.99),
                    n_steps=config.get('n_steps', 2048),
                    ent_coef=config.get('ent_coef', 0.01),
                    learning_rate=float(config.get('learning_rate', 3e-4)),
                    vf_coef=config.get('vf_coef', 0.5),
                    max_grad_norm=config.get('max_grad_norm', 0.5),
                    gae_lambda=config.get('gae_lambda', 0.95),
                    n_epochs=config.get('n_epochs', 10),
                    clip_range=config.get('clip_range', 0.2),
                    batch_size=config.get('batch_size', 64)
                )
                self.agents.append(agent)
                print(f"Successfully created PPO agent {i}")
            except Exception as e:
                print(f"Error creating PPO agent {i}: {e}")
                raise

        # --- Initialize tracking variables ---
        self.total_steps = 0
        
        # --- Initialize buffers for training ---
        if self.training:
            try:
                self.buffers = [agent.rollout_buffer for agent in self.agents]
            except AttributeError:
                raise AttributeError("PPO agents lack 'rollout_buffer'.")

        # Add in __init__:
        self.device = device  # Store the device

    # ==========================================================================
    # Core RL Loop
    # ==========================================================================

    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        # Initialize episode
        state, _ = env.reset()
        viz_data = self.init_visualization_data(env, state) if visualize else None
        rewards = []
        done = False
        t = 0

        while not done:
            self.total_steps += 1
            t += 1
            
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []

            for i in range(self.n_agents):
                try:
                    # Get action from policy
                    action, _ = self.agents[i].predict(state, deterministic=False)
                    actions.append(action.item() if isinstance(action, np.ndarray) else action)
                    
                    # Get value and log_prob for training
                    state_tensor = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) 
                                  for k, v in state.items() 
                                  if k in self.agents[i].observation_space.spaces}
                    action_tensor = torch.as_tensor([action], device=self.device)
                    value, log_prob, _ = self.agents[i].policy.evaluate_actions(state_tensor, action_tensor)
                    values.append(value.detach().cpu().item())
                    log_probs.append(log_prob.detach().cpu().item())
                except Exception as e:
                    # Fallback to random action if prediction fails
                    actions.append(self.agents[i].action_space.sample())
                    values.append(0.0)
                    log_probs.append(0.0)
            
            # Step environment
            try:
                next_state, step_rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
            except Exception as e:
                print(f"Error in env.step: {e}")
                next_state, step_rewards = state, [0.0] * self.n_agents
                done = True
            
            # Record rewards
            rewards.append(step_rewards)
            
            # Add experience to buffers if training
            if self.training:
                for i in range(self.n_agents):
                    try:
                        agent_reward = step_rewards[i] if isinstance(step_rewards, (list, tuple, np.ndarray)) else step_rewards
                        self.buffers[i].add(state, actions[i], agent_reward, done, values[i], log_probs[i])
                    except Exception as e:
                        pass  # Silent fail to avoid disrupting episode
            
            # Add visualization data if requested
            if visualize and viz_data:
                viz_data = self.add_visualization_data(viz_data, env, state, actions, next_state)
            
            # Update state
            state = next_state
            
            # Update models if training
            if self.training:
                self.update_models(state, done)
        
        # End of episode processing
        total_reward = np.sum(rewards)
        if log and not self.debug:
            self.log_one_episode(episode, t, rewards)
        
        self.print_terminal_output(episode, total_reward)
        
        if save_model:
            self.save_model_checkpoints(episode)
        
        if visualize and viz_data:
            viz_data['rewards'] = np.array(rewards)
            return viz_data
        
        return None

    def update_models(self, last_obs, done):
        if not self.training or self.total_steps <= self.config.get('initial_memory', 0): return
        buffer0 = self.buffers[0]
        is_buffer_ready = (hasattr(buffer0, 'pos') and hasattr(buffer0, 'buffer_size') and buffer0.pos == buffer0.buffer_size)
        if is_buffer_ready:
            for i in range(self.n_agents):
                try:
                    last_obs_tensor = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in last_obs.items() if k in self.agents[i].observation_space.spaces}
                    with torch.no_grad(): last_value = self.agents[i].policy.predict_values(last_obs_tensor).cpu().numpy()
                    self.buffers[i].compute_returns_and_advantage(last_values=last_value, dones=np.array([done]))
                    self.agents[i].train()
                except Exception as e: print(f"ERROR update agent {i}: {e}")

    def train(self, env):
        print(f"Starting training: {self.config.n_episodes} episodes...")
        for episode in range(self.config.n_episodes):
            viz_interval = self.config.get('visualize_every', 0)
            viz_ep = (viz_interval > 0 and (episode + 1) % viz_interval == 0 and not self.debug)
            if viz_ep:
                print(f"-- Vis Ep {episode+1} --")
                viz_data = self.run_one_episode(env, episode, log=False, train=False, save_model=False, visualize=True)
                if viz_data: self.visualize(env, f"{self.config.mode}_ep{episode+1}", viz_data=viz_data)
                # Optional: also train on viz episode? self.run_one_episode(env, episode, ... train=True ...)
            else:
                self.run_one_episode(env, episode, log=True, train=True, save_model=True, visualize=False)
        self.close_envs()
        print("Training complete.")

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def save_model_checkpoints(self, episode):
        save_interval = self.config.get('save_model_episode', 0)
        if save_interval > 0 and episode > 0 and episode % save_interval == 0:
            path = self.default_model_path
            os.makedirs(path, exist_ok=True)
            # print(f"Saving models ep {episode} to {path}") # Less verbose
            for i in range(self.n_agents):
                try: self.agents[i].save(os.path.join(path, f'agent_{i}_ep{episode}'))
                except Exception as e: print(f"ERROR saving agent {i}: {e}")

    def load_models(self, model_path=None):
        load_path_base = model_path if model_path is not None else self.default_model_path
        print(f"Loading models from: {load_path_base}")
        loaded_count = 0
        for i in range(self.n_agents):
             agent_zip = os.path.join(load_path_base, f'agent_{i}') + ".zip" # Simplistic path
             # Add logic here to find latest checkpoint if needed
             if os.path.exists(agent_zip):
                  try:
                      env_ref = self.agents[i].get_env()
                      self.agents[i] = PPO.load(agent_zip, env=env_ref, device=self.device)
                      if self.training: self.buffers[i] = self.agents[i].rollout_buffer # Relink buffer
                      loaded_count += 1
                  except Exception as e: print(f"Warn load agent {i}: {e}")
             # else: print(f"Warn: Model not found for agent {i} at {agent_zip}") # Less verbose
        print(f"Loaded {loaded_count}/{self.n_agents} models.")

    def print_terminal_output(self, episode, total_reward):
        print_interval = self.config.get('print_every', 10)
        if print_interval > 0 and (episode == 0 or (episode + 1) % print_interval == 0):
            print(f'Steps: {self.total_steps:<8} | Ep: {episode:<5} | Rew: {total_reward:<8.2f}')

    def log_one_episode(self, episode, t, rewards):
        """Log episode metrics to wandb"""
        try:
            total_reward = np.sum(rewards)
            wandb.log({
                'Episode': episode, 
                'Episode Length': t, 
                'Total Reward': total_reward,
                'Steps': self.total_steps
            })
        except Exception as e:
            # Silent fail to avoid disrupting training
            pass

    def init_visualization_data(self, env, state):
        viz_data = {'agents_partial_images': [], 'actions': [], 'full_images': [], 'rewards': []}
        try:
            viz_data['full_images'].append(env.render('rgb_array'))
            partial_views = [env.get_obs_render(self.get_agent_state(state, i).get('image'))
                             if hasattr(env, 'get_obs_render') and self.get_agent_state(state, i)
                             else np.zeros((64,64,3), dtype=np.uint8)
                             for i in range(self.n_agents)]
            viz_data['agents_partial_images'].append(partial_views)
        except Exception as e: pass # print(f"Warn: Init viz failed: {e}")
        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        """Add data from one step to the visualization buffer"""
        # Add actions
        viz_data['actions'].append(actions)
        
        try:
            # Add agent partial views
            partial_views = []
            for i in range(self.n_agents):
                agent_state = self.get_agent_state(state, i)
                if hasattr(env, 'get_obs_render') and agent_state and 'image' in agent_state:
                    partial_views.append(env.get_obs_render(agent_state['image']))
                else:
                    # Placeholder for missing render
                    partial_views.append(np.zeros((64, 64, 3), dtype=np.uint8))
            
            viz_data['agents_partial_images'].append(partial_views)
            
            # Add full environment image
            if hasattr(env, 'render'):
                viz_data['full_images'].append(env.render('rgb_array'))
        except Exception as e:
            # Silent fail to avoid disrupting episode
            pass
            
        return viz_data

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data or not viz_data.get('rewards'): print("Error: No data for viz."); return
        base_path = os.path.join(video_dir, self.config.experiment_name, self.config.model_name)
        frame_path = os.path.join(base_path, mode + "_frames")
        os.makedirs(frame_path, exist_ok=True)
        action_dict = {act.value: act.name for act in env.Actions} if hasattr(env, 'Actions') else {i: f'act_{i}' for i in range(self.agents[0].action_space.n)}
        traj_len = len(viz_data['rewards'])
        print(f"Generating {traj_len} frames for video: {mode}")
        for t in range(traj_len):
            # Check data existence implicitly via list index access below
            try:
                 frame_file = os.path.join(frame_path, f'frame_{t:04d}.png')
                 plot_single_frame(t, viz_data['full_images'][t], viz_data['agents_partial_images'][t],
                                   viz_data['actions'][t], viz_data['rewards'], action_dict, frame_file, self.config.model_name)
            except IndexError: print(f"Warn: Missing viz data at step {t}"); break # Stop if data missing
            except Exception as e: print(f"Error plotting frame {t}: {e}")
        video_filename = os.path.join(base_path, mode + '_trajectory')
        try: make_video(frame_path, video_filename); print(f"Video saved: {video_filename}.mp4")
        except Exception as e: print(f"Error creating video: {e}")

    def get_agent_state(self, state_dict, agent_idx):
        """Extract a single agent's observation from a multi-agent observation dict."""
        if not isinstance(state_dict, dict):
            return {}
            
        agent_state = {}
        for key, val in state_dict.items():
            try:
                # Handle different possible formats for agent-specific data
                if isinstance(val, (list, np.ndarray)) and len(val) > agent_idx:
                    # Multi-agent data with shape [n_agents, ...]
                    agent_state[key] = val[agent_idx]
                elif hasattr(val, 'shape') and len(val.shape) > 0 and val.shape[0] > agent_idx:
                    # Multi-agent tensor data
                    agent_state[key] = val[agent_idx]
                else:
                    # Shared data or invalid format - use as is
                    agent_state[key] = val
            except Exception as e:
                # If extraction fails, skip this key
                pass
        
        return agent_state

    def close_envs(self):
        try: self.env.close() #; print("Base env closed.")
        except Exception: pass
        try: self.agents[0].env.close() #; print("Agent env closed.")
        except Exception: pass

# ==========================================================================
# Notes on implementing CustomMultiGridPolicy
# ==========================================================================
"""
Implementing CustomMultiGridPolicy is more complex and requires several additional components:

1. **Custom Network Architecture**:
   - Create a specialized CNN for processing small grid-world observations
   - Implement custom feature extractors for multi-agent settings
   - Handle both image observations and direction information

2. **Observation Space Transformation**:
   - Implement SingleAgentObsWrapper to extract individual agent observations
   - Transform multi-agent observations (N, H, W, C) to single-agent (H, W, C)
   - Handle direction and other observation components appropriately

3. **Policy Implementation Flow**:
   a. Import CustomMultiGridPolicy from networks.multigrid_network
   b. Create separate environment wrappers for each agent using SingleAgentObsWrapper
   c. Initialize PPO with CustomMultiGridPolicy and agent-specific environments
   d. Configure policy_kwargs specifically for CustomMultiGridPolicy (kernel_size, fc_direction)
   e. Implement fallback mechanisms in case the custom policy fails

4. **Custom CNN Component**:
   - Implement a CNN suitable for small (5x5) grid observations
   - Handle various input formats and edge cases
   - Process combined observations from image and direction inputs

This approach provides more specialized handling of multi-agent grid environments
but comes with potential implementation challenges.
"""
