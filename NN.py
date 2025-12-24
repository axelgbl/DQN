'''
C - Define the Neural Network.

This is basically a replica of the Neural Network implemented in DQNelements_solved-1.py that was used in exercice session 3!
'''

import torch
import torch.nn as nn

class MyNetwork(nn.Module):
        
    def __init__(self, input_size, output_size, n_hidden):
        super().__init__()
        self.input_layer = nn.Linear(input_size, n_hidden)
        self.hidden_layer = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)