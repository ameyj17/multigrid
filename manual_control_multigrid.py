# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script used for debugging the multigrid environment via command line.

The env is rendered as a string so it can be used over ssh.
"""
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np

# Import needed to trigger env registration
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
      help='Name of multi-agent environment.')
  parser.add_argument(
      '--multiple_episodes', type=int, default=100,
  )
  return parser.parse_args()


def get_user_input(env):
  """Validates user keyboard input to obtain valid actions for all agents.

  Args:
    env: Instance of MultiGrid environment

  Returns:
    An array of integer actions.
  """
  max_action = max(env.Actions).value
  min_action = min(env.Actions).value

  # Print action commands for user convenience.
  print('Actions are:')
  for act in env.Actions:
    print('\t', str(act.value) + ':', act.name)

  prompt = 'Enter actions for ' + str(env.n_agents) + \
           ' agents separated by commas, or r to reset, or q to quit: '

  # Check user input
  while True:
    user_cmd = input(prompt)
    if user_cmd == 'q':
      return False

    # reset
    if user_cmd == 'r':
      return -1

    actions = user_cmd.split(',')
    if len(actions) != env.n_agents:
      print('Uh oh, you entered commands for', len(actions),
            'agents but there are', str(env.n_agents) + '. Try again?')
      continue

    valid = True
    for i, a in enumerate(actions):
      if not a.isdigit() or int(a) > max_action or int(a) < min_action:
        print('Uh oh, action', i, 'is invalid.')
        valid = False

    if valid:
      break
    else:
      print('All actions must be an integer between', min_action, 'and',
            max_action)

  return [int(a) for a in actions if a]


def main(args):
  # This code will only work with MultiGrid environments
  assert 'MultiGrid' in args.env_name

  env = gym.make(args.env_name)
  env.reset()
  reward_hist = []

  # Environment interaction loop
  while True:
    print(env)
    env.render()
    order_env = gym.wrappers.OrderEnforcing(env)
    print("env: \n" + str(dir(order_env)))
    print("action_space:       " + str(type(env.action_space)))
    print("observation_space:       " + str(env.observation_space))
    print("reward_space:       " + str(env.reward_range))
    print("#####################################################")
    actions = get_user_input(env)
    if not actions:
      return

    # Reset
    if actions == -1:
      env.reset()
      reward_hist = []
      plt.imshow(env.render('rgb_array'))
      continue

    _, rewards, done, _ = env.step(actions)
    reward_hist.append(rewards)
    plt.imshow(env.render('rgb_array'))
    print('Rewards:', rewards)
    print('Collective reward history:', reward_hist)
    print('Cumulative collective reward:', np.sum(reward_hist))

    if done:
      if args.multiple_episodes > 1:
        args.multiple_episodes -= 1
        env.reset()
        reward_hist = []
        continue
      print('Game over')
      break


if __name__ == '__main__':
  main(parse_args())
