'''
C - Implement the experience replay buffer.

This is basically a replica of the replay buffer implemented in DQNelements_solved-1.py that was used in exercice session 3!
'''

import numpy as np
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)