'''
C - Define the optimizer.
'''

import torch.optim as optim

def create_optimizer(network, learning_rate):
    return optim.Adam(network.parameters(), lr=learning_rate)