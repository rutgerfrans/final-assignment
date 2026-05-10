import numpy as np
import random
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


# compress raw game frames for better network processing
    # bottom row of image is stripped (contains dashboard)
    # grayscaling (color might not carry useful information) want to focus on shapes and edges, reduces number of channels
    # normalizing pixel values to [0, 1] for stable training
def preprocess_grayscale(obs):
    # (96, 96, 3) uint8 -> (1, 84, 96) uint8, drops bottom indicator strip
    gray = 0.2989 * obs[:84, :, 0] + 0.5870 * obs[:84, :, 1] + 0.1140 * obs[:84, :, 2]
    return gray.astype(np.uint8)[np.newaxis]

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
    def __init__(self, n_actions, in_channels=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        conv_out = self._conv_out_size(in_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _conv_out_size(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 96)
            return self.conv(dummy).view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

# Stores individual raw frames in a ring buffer instead of full stacked states.
# With stack_n=4 and 3-channel frames this uses ~8x less RAM than storing stacked states.
# push() takes one new frame per step; stacks are reconstructed at sample time.
class ReplayBuffer:
    def __init__(self, capacity, stack_n=4, frame_shape=(3, 84, 96)):
        self.capacity = capacity
        self.stack_n  = stack_n
        self._n = capacity + stack_n          # frame ring needs a small headroom
        self.frames     = np.zeros((self._n, *frame_shape), dtype=np.uint8)
        self.actions    = np.zeros(capacity, dtype=np.int64)
        self.rewards    = np.zeros(capacity, dtype=np.float32)
        self.dones      = np.zeros(capacity, dtype=np.float32)
        self.frame_ends = np.zeros(capacity, dtype=np.int32)  # last-frame index of each next_state
        self.size      = 0
        self.trans_pos = 0
        self.frame_pos = 0

    def reset_episode(self, init_frame):
        """Seed the ring with stack_n copies of the initial frame at each episode start."""
        for _ in range(self.stack_n):
            self.frames[self.frame_pos] = init_frame
            self.frame_pos = (self.frame_pos + 1) % self._n

    def push(self, new_frame, action, reward, done):
        self.frames[self.frame_pos]         = new_frame
        self.frame_ends[self.trans_pos]     = self.frame_pos
        self.actions[self.trans_pos]        = action
        self.rewards[self.trans_pos]        = reward
        self.dones[self.trans_pos]          = done
        self.frame_pos = (self.frame_pos + 1) % self._n
        self.trans_pos = (self.trans_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack(self, last_idx, shift=0):
        end = (last_idx - shift) % self._n
        return np.concatenate(
            [self.frames[(end - self.stack_n + 1 + i) % self._n] for i in range(self.stack_n)],
            axis=0,
        )

    def sample(self, batch_size, device):
        idxs        = np.random.choice(self.size, batch_size, replace=False)
        ends        = self.frame_ends[idxs]
        states      = np.stack([self._stack(e, shift=1) for e in ends])
        next_states = np.stack([self._stack(e, shift=0) for e in ends])
        return (
            torch.tensor(states,             dtype=torch.uint8, device=device).float() / 255.0,
            torch.tensor(self.actions[idxs], dtype=torch.int64, device=device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32, device=device),
            torch.tensor(next_states,        dtype=torch.uint8, device=device).float() / 255.0,
            torch.tensor(self.dones[idxs],   dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size
