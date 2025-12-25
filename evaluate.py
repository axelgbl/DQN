import gymnasium as gym
import torch
import numpy as np

network = torch.load('neural-network-1.pth')
network.eval()

env = gym.make('LunarLander-v3', render_mode="human")
state = env.reset()[0]
done, truncated = False, False
total_reward = 0

while not (done or truncated):
    env.render()
    state_tensor = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        action = network(state_tensor).argmax().item()
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    total_reward += reward

print(f"Evaluation Total Reward: {total_reward}")
env.close()