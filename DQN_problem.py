# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v3', render_mode = "human")


env.reset()

# Parameters
N_episodes = 100                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Close environment
env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

'''
C - Implement DQN and solve the problem.
'''

N_EPISODES = 100 # Number of episodes between 100-1000
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
GAMMA = 0.99
EPSILON = 0.99
EPSILON_MIN = 0.05
EPSILON_DECAY = int(0.9*N_EPISODES) # Decay over Z ~ 90%-95% of episodes
BATCH_SIZE = 64 # Size of the training batch
BUFFER_SIZE = 20000 # Size of the experience replay buffer between 5000-30000
LEARNING_RATE = 1e-3 # Learning rate between 1e-3 - 1e-4
MAX_STEPS = 200
TARGET_UPDATE_FREQ = int(BUFFER_SIZE/BATCH_SIZE) # C ~ L/N
CLIPPING_VALUE = 1 # Norm gradient clipping value between 0.5-2
N_HIDDEN = 64 # Number of neurons per hidden layer between 8-128
N_EP_RUNNING_AVERAGE = 50

env = gym.make('LunarLander-v3')

env.reset()

n_actions = env.action_space.n
dim_state = len(env.observation_space.high)

episode_reward_list = []
episode_number_of_steps = []

agent = DQNAgent(n_actions, dim_state, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BATCH_SIZE, BUFFER_SIZE, LEARNING_RATE, MAX_STEPS, TARGET_UPDATE_FREQ, CLIPPING_VALUE, N_HIDDEN)

for i in EPISODES:

    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not (done or truncated):

        action = agent.forward(state)
        next_state, reward, done, truncated, _ = env.step(action)

        agent.store_experience(state, reward, next_state, done or truncated)
        agent.backward()

        total_episode_reward += reward
        state = next_state
        t+= 1

    agent.decay_epsilon()
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    if running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1] >= 200:
        print(f"Solved in episode {i}!")
        break

    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
        running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

torch.save(agent.network.state_dict(), 'dqn_lunar_lander.pth')

env.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_EPISODES+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_EPISODES+1)], running_average(
    episode_reward_list, N_EP_RUNNING_AVERAGE), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_EPISODES+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_EPISODES+1)], running_average(
    episode_number_of_steps, N_EP_RUNNING_AVERAGE), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()