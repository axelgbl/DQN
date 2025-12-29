""" Question f """

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NN import MyNetwork

VALUES = 50

# Load the trained network
network = torch.load('neural-network-1.pth', weights_only=False)
network.eval()

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
            q_values = network(state_tensor)
        
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