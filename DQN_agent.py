# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ExperienceReplayBuffer, Experience
from NN import MyNetwork
from optimizer import create_optimizer

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class DQNAgent(Agent):
    
    def __init__(self, n_actions, dim_state, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, buffer_size, learning_rate, max_steps, target_update_freq, clipping_value, n_hidden):
        super().__init__(n_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_steps = max_steps
        self.clipping_value = clipping_value

        self.steps = 1
        self.episodes = 1

        self.buffer = ExperienceReplayBuffer(buffer_size)
        self.network = MyNetwork(dim_state, n_actions, n_hidden)
        self.target_network = MyNetwork(dim_state, n_actions, n_hidden)
        self.optimizer = create_optimizer(self.network, learning_rate)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max-self.epsilon_min)*(self.episodes - 1)/(self.epsilon_decay-1))
        self.episodes += 1
    
    def forward(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            self.last_action = np.random.randint(0, self.n_actions)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                self.last_action = self.network(state_tensor).argmax().item()
        return self.last_action

    def backward(self):
        if len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
        
            q_values = self.network(states).gather(1, actions).squeeze()

            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, targets)
                
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.clipping_value)
            self.optimizer.step()

            self.steps += 1
            if self.steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
