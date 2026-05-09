import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque

# compress raw game frames for better network processing
    # bottom row of image is stripped (contains dashboard)
    # grayscaling (color might not carry useful information) want to focus on shapes and edges, reduces number of channels
    # normalizing pixel values to [0, 1] for stable training
def preprocess(obs):
    # (96, 96, 3) uint8 -> (84, 96) float32 grayscale, drops bottom indicator strip
    gray = 0.2989 * obs[:84, :, 0] + 0.5870 * obs[:84, :, 1] + 0.1140 * obs[:84, :, 2]
    return gray.astype(np.float32) / 255.0

def preprocess_without_graysscale(obs):
    # (96, 96, 3) uint8 -> (3, 84, 96) uint8, drops bottom indicator strip
    return obs[:84].transpose(2, 0, 1).copy()

# this class keeps a sliding window of the last 4 frames so the agent can perceive motion.
class FrameStack:
    def __init__(self, n):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, frame):
        for _ in range(self.n):
            self.frames.append(frame)

    def push(self, frame):
        self.frames.append(frame)

    def get(self):
        return np.concatenate(self.frames, axis=0)  # (n*c, 84, 96)

# DQN model, takes stack of 4 frames as input and outputs Q-values for each action
# Based on a standard convolutional architecture for processing image inputs, 
# with 3 convolutional layers followed by 2 fully connected layers.
# Made with PyTorch's nn.Module. https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        conv_out = self._conv_out_size()
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _conv_out_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 84, 96)
            return self.conv(dummy).view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

# replay buffer for storing and sampling transitions during training
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.uint8, device=device).float() / 255.0,
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.uint8, device=device).float() / 255.0,
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)
