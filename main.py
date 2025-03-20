import argparse
import random
import torch
import numpy as np
import wandb
from multiagent_metacontroller import MetaController
import utils
import gym

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
        help='Name of environment.')
    parser.add_argument(
        '--mode', type=str, default='ppo',
        help="Name of experiment. Can be 'ppo'")
    parser.add_argument(
        '--debug', action=argparse.BooleanOptionalAction,
        help="If used will disable wandb logging.")
    parser.add_argument(
        '--seed', type=int, default=1,
        help="Random seed.")
    parser.add_argument(
        '--keep_training', action=argparse.BooleanOptionalAction,
        help="If used will continue training from previous checkpoint.")
    parser.add_argument(
        '--visualize', action=argparse.BooleanOptionalAction,
        help="If used will visualize agent behavior without training.")
    parser.add_argument(
        '--video_dir', type=str, default='videos',
        help="Name of location to store videos.")
    parser.add_argument(
        '--load_checkpoint_from',  type=str, default=None,
        help="Path to find model checkpoints to load")
    parser.add_argument(
        '--wandb_project', type=str, default='multigrid_expert_3',
        help="Name of wandb project.")
    parser.add_argument(
        '--n_episodes', type=int, default=100000,
        help="Number of episodes to train.")
    parser.add_argument(
        '--decentralized_training', action=argparse.BooleanOptionalAction, default=False,
        help="If used, will train agents independently rather than with coordinated actions.")

    return parser.parse_args()

def get_metacontroller_class(config):
    """
        - Input : Observation() --> (state, reward)
        - Output: Action() (a1, a2, a3) --> to MultiGrid
    """
    return MetaController
    #raise NotImplementedError("Implement and import a MetaController class!")

def initialize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: "+ str(device))
    
    # Generate parameters
    config = utils.generate_parameters(
        mode=args.mode, 
        domain=args.env_name, 
        debug=(args.debug or args.visualize), 
        seed=args.seed, 
        wandb_project=args.wandb_project
    )
    
    # Add additional parameters from command line
    config.n_episodes = args.n_episodes
    config.decentralized_training = args.decentralized_training

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = utils.make_env(config)
    metacontroller_class = get_metacontroller_class(config)

    return device, config, env, metacontroller_class

def main(args):
    # Validate wandb project if logging
    if not args.debug and not args.visualize:
        if not args.wandb_project:
            print('ERROR: when logging to wandb, must specify a valid wandb project.')
            exit(1)

        current_wandb_projects = ['multigrid_expert_3']
        if str(args.wandb_project) not in current_wandb_projects:
            print('ERROR: wandb project not in current projects. '
                  'Change the project name or add your new project to the current projects in current_wandb_projects. '
                  'Current projects are:', current_wandb_projects)
            exit(1)

    # Initialize components
    device, config, env, metacontroller_class = initialize(args)
    print("Initialization successful")

    # Visualization mode
    if args.visualize:
        agent = metacontroller_class(config, env, device, training=False)
        agent.load_models(model_path=args.load_checkpoint_from)
        agent.visualize(env, args.mode, args.video_dir)
        print('A video of the trained policies being tested in the environment'
              'has been generated and is located in', args.video_dir)
        exit(0)
    
    # Training mode
    print("Setting up metacontroller agent")
    agent = metacontroller_class(config, env, device, debug=args.debug)

    if args.keep_training and args.load_checkpoint_from:
        print(f"Loading models from {args.load_checkpoint_from}")
        agent.load_models(model_path=args.load_checkpoint_from)

    print(f"Starting training for {config.n_episodes} episodes")
    agent.train(env)
    print("Training complete")

if __name__ == '__main__':
    main(parse_args())