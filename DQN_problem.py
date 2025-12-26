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
from replay_buffer import Experience
import warnings
from mpl_toolkits.mplot3d import Axes3D

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
N_episodes = 200                             # Number of episodes
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
ax[0].set_title('Total Reward with Random Agent')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps with Random Agent')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

"""
'''
C - Implement DQN and solve the problem.
'''

N_EPISODES = 200 # Number of episodes between 100-1000 # I put 200 because the average episodic reward never reach 50 with only 100 episodes
GAMMA = 0.99
EPSILON = 0.99
EPSILON_MIN = 0.05
EPSILON_DECAY = int(0.9*N_EPISODES) # Decay over Z ~ 90%-95% of episodes
# Important Note: The variable EPSILON_DECAY is here different from the one in the DQNelements_solved-1.py file that was used during exercice session 3. It is here used to refer to the number Z of episodes epsilon is decayed!
BATCH_SIZE = 64 # Size of the training batch between 4-128
BUFFER_SIZE = 20000 # Size of the experience replay buffer between 5000-30000
LEARNING_RATE = 1e-3 # Learning rate between 1e-3 - 1e-4
MAX_STEPS = 1000
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

EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)

for i in EPISODES:

    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not (done or truncated or t >= MAX_STEPS):

        action = agent.forward(state)
        next_state, reward, done, truncated, _ = env.step(action)

        agent.buffer.append(Experience(state, action, reward, next_state, done or truncated))
        agent.backward()

        total_episode_reward += reward
        state = next_state
        t+= 1

    if i<agent.epsilon_decay: agent.decay_epsilon()
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    if running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1] >= 50:
        print(f"Problem solved in episode {i}!")
        break

    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
        running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

torch.save(agent.network, 'neural-network-1.pth')

env.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
    episode_reward_list, N_EP_RUNNING_AVERAGE), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward with DQN')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
    episode_number_of_steps, N_EP_RUNNING_AVERAGE), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps with DQN')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()


'''
Model with buffer size of 5000
'''

N_EPISODES = 200 # Number of episodes between 100-1000 
GAMMA = 0.99
EPSILON = 0.99
EPSILON_MIN = 0.05
EPSILON_DECAY = int(0.9*N_EPISODES) # Decay over Z ~ 90%-95% of episodes
# Important Note: The variable EPSILON_DECAY is here different from the one in the DQNelements_solved-1.py file that was used during exercice session 3. It is here used to refer to the number Z of episodes epsilon is decayed!
BATCH_SIZE = 64 # Size of the training batch between 4-128
BUFFER_SIZE = 5000 # Size of the experience replay buffer between 5000-30000
LEARNING_RATE = 1e-3 # Learning rate between 1e-3 - 1e-4
MAX_STEPS = 1000
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

EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)

for i in EPISODES:

    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not (done or truncated or t >= MAX_STEPS):

        action = agent.forward(state)
        next_state, reward, done, truncated, _ = env.step(action)

        agent.buffer.append(Experience(state, action, reward, next_state, done or truncated))
        agent.backward()

        total_episode_reward += reward
        state = next_state
        t+= 1

    if i<agent.epsilon_decay: agent.decay_epsilon()
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    if running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1] >= 50:
        print(f"Problem solved in episode {i}!")
        break

    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
        running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

env.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
    episode_reward_list, N_EP_RUNNING_AVERAGE), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward with buffer size of 5000')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
    episode_number_of_steps, N_EP_RUNNING_AVERAGE), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps with buffer size of 5000')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

'''
Model with buffer size of 30000
'''

N_EPISODES = 200 # Number of episodes between 100-1000 
GAMMA = 0.99
EPSILON = 0.99
EPSILON_MIN = 0.05
EPSILON_DECAY = int(0.9*N_EPISODES) # Decay over Z ~ 90%-95% of episodes
# Important Note: The variable EPSILON_DECAY is here different from the one in the DQNelements_solved-1.py file that was used during exercice session 3. It is here used to refer to the number Z of episodes epsilon is decayed!
BATCH_SIZE = 64 # Size of the training batch between 4-128
BUFFER_SIZE = 30000 # Size of the experience replay buffer between 5000-30000
LEARNING_RATE = 1e-3 # Learning rate between 1e-3 - 1e-4
MAX_STEPS = 1000
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

EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)

for i in EPISODES:

    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not (done or truncated or t >= MAX_STEPS):

        action = agent.forward(state)
        next_state, reward, done, truncated, _ = env.step(action)

        agent.buffer.append(Experience(state, action, reward, next_state, done or truncated))
        agent.backward()

        total_episode_reward += reward
        state = next_state
        t+= 1

    if i<agent.epsilon_decay: agent.decay_epsilon()
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    if running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1] >= 50:
        print(f"Problem solved in episode {i}!")
        break

    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
        running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

env.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
    episode_reward_list, N_EP_RUNNING_AVERAGE), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward with buffer size of 30000')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
    episode_number_of_steps, N_EP_RUNNING_AVERAGE), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps with buffer size of 30000')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

"""

""" Question f """

VALUES = 50

# Load the trained network
agent.network = torch.load('neural-network-1.pth', weights_only=False)
agent.network.eval()

y_values = np.linspace(0, 1.5, VALUES)
omega_values = np.linspace(-np.pi, np.pi, VALUES)
Y, Omega = np.meshgrid(y_values, omega_values)

Q_max = np.zeros_like(Y)
Q_argmax = np.zeros_like(Y)

for i in range(len(y_values)):
    for j in range(len(omega_values)):
        state = np.array([0, y_values[i], 0, 0, omega_values[j], 0, 0, 0], dtype=np.float32)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.network(state_tensor)
        
        Q_max[j, i] = q_values.max().item()
        Q_argmax[j, i] = q_values.argmax().item()


# Create the plots

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y, Omega, Q_max, cmap='viridis')

ax.set_xlabel('Height (y)')
ax.set_ylabel('Angle (ω)')
ax.set_zlabel('max_a Q(s(y,ω),a)')
ax.set_title('Maximum Q-value for restricted state space')

cbar = plt.colorbar(surf, ax=ax, label='Q-value', shrink=0.5)
plt.show()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y, Omega, Q_argmax, cmap='viridis')

ax.set_xlabel('Height (y)')
ax.set_ylabel('Angle (ω)')
ax.set_zlabel('Optimal Action argmax_a Q(s(y,ω),a)')
ax.set_title('Optimal Action for restricted state space')

cbar = plt.colorbar(surf, ax=ax, label='Action', shrink=0.5)
plt.show()

