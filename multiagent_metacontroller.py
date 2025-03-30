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
#from networks import MultiInputPolicy

from utils import plot_single_frame, make_video, extract_mode_from_path

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import DictRolloutBuffer

from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces

class GymToGymnasiumDictWrapper(gym.Wrapper):
    """
    Converts an old Gym environment with `gym.spaces.Dict` observation space
    to use `gymnasium.spaces.Dict`, making it compatible with Stable-Baselines3. 
    alternative wrapper: gymnasium.env 
    """
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym_spaces.Dict), \
            "This wrapper only supports Dict observation spaces."

        self.observation_space = gymnasium_spaces.Dict({
            key: self._convert_space(space) for key, space in env.observation_space.spaces.items()
        })

        self.action_space = self._convert_space(env.action_space)

    def _convert_space(self, space):
        if isinstance(space, gym_spaces.Box):
            return gymnasium_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gym_spaces.Discrete):
            return gymnasium_spaces.Discrete(n=space.n, start=space.start)
        elif isinstance(space, gym_spaces.Tuple):
            return gymnasium_spaces.Tuple(tuple(self._convert_space(s) for s in space.spaces))
        elif isinstance(space, gym_spaces.Dict):
            return gymnasium_spaces.Dict({key: self._convert_space(s) for key, s in space.spaces.items()})
        else:
            return space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym_spaces.Discrete(7)

    def step(self, action):
        action = int(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward[0], done, info


class MultiAgentPPOController():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.debug = debug

        self.default_model_path = os.path.join('models', config.experiment_name, config.model_name)

        # Flag to enable modeling of other agents (currently affects visualization only)
        self.model_others = config.get("model_others", False) 

        self.n_agents = env.n_agents
        #print("Number of agents: ", self.n_agents)

        # Initialize PPO agents for each agent in the environment
        self.agents = [PPO("MultiInputPolicy", DiscreteActionWrapper(self.env), verbose=1, device=self.device)
                       for i in range(self.n_agents)]
        
        # Configure logging for each agent
        for i in range(self.n_agents):
            self.agents[i]._logger = configure(folder="logs/", format_strings=["stdout"])

        self.total_steps = 0

        #print(type(self.env.observation_space))

        if self.training:
            # Note: Alternative buffer management strategies could be implemented here.
            # The current approach uses the default RolloutBuffer from each PPO agent.
            self.buffers = [
                self.agents[i].rollout_buffer
                for i in range(self.n_agents)
            ]
    
    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        # Reset environment and initialize episode variables
        try:
            state = env.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return None # Cannot proceed if reset fails

        if visualize:
            viz_data = self.init_visualization_data(env, state)

        rewards = []
        done = False
        t = 0

        # Main episode loop
        while not done:
            self.total_steps += 1
            t += 1

            actions = []
            log_probs = []

            # Collect actions and log probabilities for each agent
            for i in range(self.n_agents):
                try:
                    # Predict action using the agent's policy
                    action, _ = self.agents[i].predict(state, deterministic=False)
                    actions.append(action)

                    # Prepare state and action tensors for policy evaluation
                    state_torch = {key: torch.tensor(val, dtype=torch.float32, device=self.device).unsqueeze(0) for key, val in state.items()}
                    action_tensor = torch.tensor(action, device=self.device).unsqueeze(0)

                    # Evaluate action to get log probability
                    _, log_prob, _ = self.agents[i].policy.evaluate_actions(state_torch, action_tensor)
                    log_probs.append(log_prob.detach()) # Detach to prevent gradients from flowing back here

                except Exception as e:
                    print(f"Error during action prediction/evaluation for agent {i}: {e}")
                    # Handle error, e.g., use a default action or skip step
                    actions.append(self.env.action_space.sample()) # Example: random action
                    log_probs.append(torch.tensor(0.0, device=self.device)) # Example: zero log prob


            # Step the environment with the collected actions
            try:
                next_state, step_rewards, done_new, infos = env.step(actions)
                rewards.append(step_rewards)
            except Exception as e:
                print(f"Error during environment step: {e}")
                done = True # End episode if step fails
                continue

            # Add experience to each agent's buffer
            for i in range(self.n_agents):
                try:
                    # Prepare state tensor for value prediction
                    state_torch = {key: torch.tensor(val, dtype=torch.float32, device=self.device).unsqueeze(0) for key, val in state.items()}
                    # Predict value of the current state
                    value = self.agents[i].policy.predict_values(state_torch).detach()
                    
                    # Add collected experience to the buffer
                    # Ensure correct types/devices are handled by the buffer implementation
                    self.buffers[i].add(state, actions[i], step_rewards[i], done, value, log_probs[i])
                except Exception as e:
                     print(f"Error adding to buffer for agent {i}: {e}")
                     # Decide how to handle buffer errors, e.g., skip this transition

            # Update visualization data if enabled
            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, state, actions, next_state)
            
            # Move to the next state
            state = next_state
            done = done_new # Update done flag based on environment step

            # Update models periodically based on buffer contents
            self.update_models(state, done)


        # --- End of episode ---

        # Log results and save model checkpoints
        if log: self.log_one_episode(episode, t, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        if save_model: self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data
        return None # Return None if not visualizing

    def save_model_checkpoints(self, episode):
        # Save agent models periodically
        if episode > 0 and episode % self.config.save_model_episode == 0:
            for i in range(self.n_agents):
                try:
                    save_path = os.path.join(self.default_model_path, '_agent_' + str(i))
                    self.agents[i].save(save_path)
                    # print(f"Saved model for agent {i} at episode {episode} to {save_path}")
                except Exception as e:
                    print(f"Error saving model for agent {i}: {e}")


    def print_terminal_output(self, episode, total_reward):
        # Print progress to terminal
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        # Initialize data structure for storing visualization frames
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None # Only used if self.model_others is True
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            # Add initial action predictions if modeling others
            predicted_actions = [self.get_action_predictions(state)]
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        # Append data for the current step to visualization structure
        viz_data['actions'].append(actions)
        # Render partial observations for each agent
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image']) for i in range(self.n_agents)])
        # Render the full environment view
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            # Add action predictions for the next state if modeling others
            viz_data['predicted_actions'].append(self.get_action_predictions(next_state))
        return viz_data
        
    def update_models(self, state, dones):
        # Update agent models if enough steps have passed and it's an update interval
        if self.total_steps > self.config.initial_memory:
            if self.total_steps % self.config.update_every == 0: 
                for i in range(self.n_agents):
                    try:
                        # Prepare state tensor for final value prediction
                        state_torch = {key: torch.tensor(val, dtype=torch.float32, device=self.device).unsqueeze(0) for key, val in state.items()}
                        # Predict value of the final state in the rollout
                        value = self.agents[i].policy.predict_values(state_torch).detach()
                        # Compute returns and advantages using the collected rollout data
                        self.buffers[i].compute_returns_and_advantage(last_values=value, dones=np.array([dones]))
                        # Perform the PPO training step
                        self.agents[i].train()
                        # Reset the buffer for the next rollout collection
                        self.buffers[i].reset()
                    except Exception as e:
                        print(f"Error updating model for agent {i}: {e}")
                        # Optionally, try to reset buffer or skip update if train fails
                        try:
                            self.buffers[i].reset() # Attempt to reset buffer even if train failed
                        except Exception as reset_e:
                            print(f"Error resetting buffer for agent {i} after update error: {reset_e}")

    
    def train(self, env):
        # Main training loop
        print("Training...")
        for episode in range(self.config.n_episodes):
            # Optionally visualize training progress
            if (episode + 1) % self.config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                if viz_data: # Ensure viz_data is not None (e.g., if run_one_episode failed)
                    self.visualize(env, self.config.mode + '_training_step' + str(episode), 
                                   viz_data=viz_data)
                else:
                    print(f"Skipping visualization for episode {episode} due to error in run_one_episode.")
            else:
                # Run a standard training episode
                self.run_one_episode(env, episode)

        # Close environment after training completion
        try:
            env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        # Generate visualization video from an episode run
        if not viz_data:
            # Run a new episode specifically for visualization if no data provided
            print("Running new episode for visualization...")
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            if not viz_data:
                 print("Failed to generate visualization data.")
                 return # Cannot visualize if episode run failed
            try:
                 env.close() # Close env if it was opened just for this visualization
            except Exception as e:
                 print(f"Error closing env after visualization run: {e}")


        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Create directory for video frames if it doesn't exist
        if not os.path.exists(video_path):
            try:
                os.makedirs(video_path)
            except OSError as e:
                print(f"Error creating video directory {video_path}: {e}")
                return # Cannot proceed without directory


        # Map action indices to human-readable names
        action_dict = {act.value: act.name for act in env.Actions}

        traj_len = len(viz_data.get('rewards', []))
        if traj_len == 0:
            print("Warning: No rewards data found for visualization.")
            # Potentially add checks for other viz_data keys

        # Generate and save each frame of the visualization
        for t in range(traj_len):
            try:
                self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
                # print('Frame {}/{}'.format(t + 1, traj_len)) # User-friendly 1-based indexing
            except KeyError as e:
                print(f"Error generating frame {t}: Missing key {e} in viz_data.")
                continue # Skip this frame
            except Exception as e:
                print(f"Error generating frame {t}: {e}")
                continue # Skip this frame


        # Compile frames into a video
        try:
            print(f"Attempting to create video at: {video_path}/{mode}_trajectory_video")
            make_video(video_path, mode + '_trajectory_video')
            print("Video creation successful.")
        except Exception as e:
            print(f"Error creating video: {e}")

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        # Helper function to plot a single frame using utility function
        # Ensure all required keys exist in viz_data before accessing
        required_keys = ['full_images', 'agents_partial_images', 'actions', 'rewards']
        if not all(key in viz_data for key in required_keys):
            raise KeyError(f"viz_data missing one or more required keys for frame {t}")
        if t >= len(viz_data['full_images']) or \
           t >= len(viz_data['agents_partial_images']) or \
           t >= len(viz_data['actions']):
             raise IndexError(f"Index {t} out of bounds for viz_data lists.")

        plot_single_frame(t, 
                          viz_data['full_images'][t], 
                          viz_data['agents_partial_images'][t], 
                          viz_data['actions'][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config.model_name, 
                          # Optional arguments based on self.model_others flag
                          # predicted_actions=viz_data.get('predicted_actions'), 
                          # all_actions=viz_data['actions'] # If needed by plot_single_frame
                          )

    def load_models(self, model_path=None):
        # Load pre-trained models for each agent
        load_path_base = model_path if model_path is not None else self.default_model_path
        
        for i in range(self.n_agents):
            agent_model_path = os.path.join(load_path_base, '_agent_' + str(i))
            try:
                print(f"Loading model for agent {i} from {agent_model_path}...")
                self.agents[i] = PPO.load(agent_model_path, env=DiscreteActionWrapper(self.env), device=self.device)
                print(f"Successfully loaded model for agent {i}.")
                # Re-configure logger after loading
                self.agents[i]._logger = configure(folder="logs/", format_strings=["stdout"]) 
            except FileNotFoundError:
                print(f"Error: Model file not found for agent {i} at {agent_model_path}")
            except Exception as e:
                print(f"Error loading model for agent {i} from {agent_model_path}: {e}")


    def log_one_episode(self, episode, t, rewards):
        # Log episode results using wandb
        try:
            wandb.log({'Episode': episode, 'Steps': t, 'Total Reward': np.sum(rewards)})
        except Exception as e:
            print(f"Error logging to wandb: {e}")

    def get_agent_state(self, state, agent_idx):
        # Extract the observation dictionary for a specific agent
        try:
            return {key: val[agent_idx] for key, val in state.items()}
        except IndexError:
             print(f"Error: agent_idx {agent_idx} out of range for state.")
             return {} # Return empty dict or handle error appropriately
        except KeyError as e:
             print(f"Error: Key {e} not found in state dictionary.")
             return {} # Return empty dict or handle error appropriately
# ==========================================================================
# Notes on implementing CustomMultiGridPolicy
# ==========================================================================
"""
Implementing CustomMultiGridPolicy is a bit complex and requires several additional components:

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
