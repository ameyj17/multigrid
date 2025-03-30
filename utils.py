import gym
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from moviepy.editor import *
import numpy as np
import os
import random
import seaborn as sns
import torch
import wandb
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def merge_configs(update, default):
    if isinstance(update,dict) and isinstance(default,dict):
        for k,v in default.items():
            if k not in update:
                update[k] = v
            else:
                update[k] = merge_configs(update[k],v)
    return update

def make_env(config):
    if 'MultiGrid' in config.domain:
        from envs import gym_multigrid
        from envs.gym_multigrid import multigrid_envs
        env = gym.make(config.domain)
    else:
        raise NotImplementedError
    return env

def argmax_2d_index(arr):
    assert len(arr.shape) == 2
    best_2d_index = (arr==torch.max(arr)).nonzero()
    if best_2d_index.shape[0] > 1:  # Handle case with multiple equal maxs
        # Randomly select from multiple equal maxes
        best_2d_index = best_2d_index[random.randrange(best_2d_index.shape[0]),:]
    return best_2d_index.squeeze()

def process_state(state, observation_shape):
    if len(observation_shape) == 3:
        state = torch.tensor(state)
        state = state.transpose(0, 2).transpose(1, 2)
        state = state.float().unsqueeze(0)  # swapped RGB dimension to come first
    return state

def generate_parameters(mode, domain, debug=False, seed=None, with_expert=None, wandb_project=None):
    os.environ["WANDB_MODE"] = "online"

    # config parameters
    config_default = yaml.safe_load(open("config/default.yaml", "r"))
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))
    config_mode = yaml.safe_load(open("config/mode/" + mode + ".yaml", "r"))

    # override default random seed 
    if seed:
        config_default['seed'] = seed

    config_default['experiment_name'] = 'MultiGrid'  # TODO: change me

    # Merge configs
    config_with_domain = merge_configs(config_domain, config_default)
    config = dotdict(merge_configs(config_mode, config_with_domain))
    print("merge_configs: " + str(config))

    # Print n_episodes from config - add this for debugging
    if 'n_episodes' in config:
        print(f"n_episodes from YAML config: {config['n_episodes']}")
    else:
        print("Warning: n_episodes not found in YAML config")

    # Create a wandb config dictionary - DO NOT exclude n_episodes
    wandb_config_dict = {k: v for k, v in config.items() if k not in ['decentralized_training']}
    
    if debug:
        # Disable weights and biases logging during debugging
        print("debug: " + str(type(debug)))
        print('Debug selected, disabling wandb')
        wandb.init(project = wandb_project + '-' + domain, config=wandb_config_dict, 
            mode='disabled')
    else:
        print("Initiating wandb project")
        wandb.init(project = wandb_project + '-' + domain, config=wandb_config_dict)
    
    path_configs = {'model_name': config.mode + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'load_model_path': (config.load_model_start_path or "") + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'wandb_project': wandb_project + '-' + config.domain}
    wandb.config.update(path_configs)

    print("CONFIG")
    print(wandb.config)
    
    # Copy over any important config values that may have been lost
    if 'n_episodes' in config and 'n_episodes' not in wandb.config:
        wandb.config.update({'n_episodes': config.n_episodes}, allow_val_change=True)
        print(f"Added n_episodes={config.n_episodes} to wandb.config")

    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")

    # set all other train/ metrics to use this step
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    wandb.run.name = config.model_name

    # Make sure we always return both the wandb config and our original config
    # Return the original merged config instead of just wandb.config
    return config


def plot_single_frame(t, full_image, agents_partial_images, actions, rewards, action_dict, video_path, model_name, predicted_actions=None):
    """
    Plot a single frame of the trajectory visualization.
    """
    try:
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Seaborn palette.
        sns.set()
        color_palette = sns.palettes.color_palette()

        # Hardcoded plot settings
        linewidth = 1.25
        ms_current = 9
        xlabelpad = 9
        ylabelpad = 10

        # Determine variables
        n_agents = len(actions)
        max_val = np.max(full_image)

        # Create figure
        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        total_subplots_horizontal = 2 + n_agents
        total_subplots_vertical = 3
        gs = GridSpec(total_subplots_vertical, total_subplots_horizontal, figure=fig)
         
        # Create sub plots as grid
        full_obs_ax = fig.add_subplot(gs[:2, :2])  # Overall view fig is 2x2 (larger)
        collective_reward_ax = fig.add_subplot(gs[2,:2]) 
        agents_obs_axes = []
        agents_rewards_axes = []
        for i in range(n_agents):
            agents_obs_axes.append(fig.add_subplot(gs[0, i+2]))
            agents_rewards_axes.append(fig.add_subplot(gs[2, i+2]))

        # Determine grid proportions
        full_obs_proportion = 2.0 / total_subplots_horizontal
        agent_proportion = 1.0 / total_subplots_horizontal

        # Plot shared obervation in top left
        full_obs_ax.imshow(full_image, interpolation='none')
        full_obs_ax.set_title('Full environment state')
        full_obs_ax.grid(False)

        # Plot individual agents' observations across top right
        for i in range(n_agents):
            agents_obs_axes[i].imshow(agents_partial_images[i], interpolation='none')
            agents_obs_axes[i].set_title('Agent' + str(i) + ' partial obs')
            agents_obs_axes[i].grid(False)

        # Plot collective return bottom left
        collective_return = np.sum(rewards,axis=1)
        cum_collective_return = np.cumsum(collective_return)
        steps = np.arange(len(cum_collective_return))
        collective_reward_ax.plot(steps, cum_collective_return, color=color_palette[0], lw=linewidth)
        if t > 0:
            collective_reward_ax.plot(t, cum_collective_return[t - 1], 'o', ms=ms_current, 
                  mfc=color_palette[0], mew=0)
            
            # Write the reward for previous timestep
            s = 'R_t={}: {}'.format(t-1, collective_return[t-1])
            collective_reward_ax.text(0.1, .85, s, fontsize=10,
                                      horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)
        collective_reward_ax.set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
        collective_reward_ax.set_ylabel('Collective return', fontsize=10, labelpad=ylabelpad)

        # Write the reward for current timestep
        s = 'R_t={}: {}'.format(t, collective_return[t])
        collective_reward_ax.text(0.1, 0.7, s, fontsize=10, 
                                  horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)

        # Plot individual agent returns and actions
        for i in range(n_agents):
            # Cumulative return graphs across bottom right
            cum_return = np.cumsum(rewards[:,i])
            agents_rewards_axes[i].plot(steps, cum_return, color=color_palette[0], lw=linewidth)
            if t > 0:
                agents_rewards_axes[i].plot(t, cum_return[t - 1], 'o', ms=ms_current, mfc=color_palette[0], mew=0)
            agents_rewards_axes[i].set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
            agents_rewards_axes[i].set_ylabel('Agent' + str(i) + ' return', fontsize=10, labelpad=ylabelpad)

            # Write the current action and rewards in the space between subplots
            text_horizontal_loc = full_obs_proportion + agent_proportion * i + agent_proportion * 0.2
            if predicted_actions is not None:
                text_vertical_loc = 0.75
            else:
                text_vertical_loc = 0.65
            act_text = 'a^{}_t={}: {}'.format(i, t, action_dict[int(actions[i])])  # action
            fig.text(text_horizontal_loc, text_vertical_loc, act_text, fontsize=10)
            r_text = 'R_t={}: {}'.format(t, rewards[t, i])
            fig.text(text_horizontal_loc, text_vertical_loc-0.1, r_text, fontsize=10)
            if t > 0:
                r_prev_text = 'R_t={}: {}'.format(t-1, rewards[t-1, i])
                fig.text(text_horizontal_loc, text_vertical_loc-0.05, r_prev_text, fontsize=10)

        filename = '{}_{:05d}.png'.format(model_name, t)
        fig_path = os.path.join(video_path, filename)
        plt.savefig(fig_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_single_frame for frame {t}: {e}")
        # Make sure to close any open figures even if an error occurs
        plt.close('all')

def make_video(video_path, video_name='trajectory_video', frame_rate=10, img_extension='.png'):
    """Create video from frames with proper error handling"""
    image_files = [os.path.join(video_path, img) for img in os.listdir(video_path) if img.endswith(img_extension)]
    image_files.sort()
    
    # Inform user about frame count
    print(f"Creating video with {len(image_files)} frames")
    
    if len(image_files) == 0:
        print("Warning: No frames found for video creation")
        return
    
    try:
        clips = [ImageClip(img).set_duration(1/frame_rate) for img in image_files]
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(os.path.join(video_path, video_name + '.mp4'), fps=frame_rate)
        print(f"Video successfully created: {os.path.join(video_path, video_name + '.mp4')}")
    except Exception as e:
        print(f"Error creating video: {e}")

def print_network_params(net):
    for name, p in net.named_parameters(): 
        print(name, p.data.shape)

def extract_mode_from_path(str):
    for mode in ['dqn', 'bcaux', 'basis', 'psiphi', 'copy']:
        if mode in str:
            return mode 
    assert False, 'No known mode in path ' + str