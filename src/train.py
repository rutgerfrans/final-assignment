import glob
import os
import random
import torch
import torch.nn as nn
import gymnasium as gym
from src.model import DQN, ReplayBuffer, FrameStack, preprocess_without_graysscale
from src.config import (ROOT, WEIGHTS_DIR, EPS_START, EPS_END, EPS_DECAY, TRAIN_START,
                        TARGET_UPDATE, SAVE_EVERY, MAX_EPISODES, GAMMA,
                        LR, BATCH_SIZE, BUFFER_SIZE, STACK_N, PRE_TRAINED)

def make_env():
    return gym.make("CarRacing-v3", continuous=False)

# Epsilon-greedy action selection
def select_action(state, eps, net, device):
    if random.random() < eps:
        return random.randint(0, 4)
    with torch.no_grad():
        s = torch.tensor(state[None], dtype=torch.float32, device=device)
        return net(s).argmax(dim=1).item()

# Compute the MSE loss between the Q-values predicted by the policy network 
# and the target Q-values computed using the target network.
# Done using Torch's no_grad() and Torch's nn.functional.mse_loss for the loss calculation.
# https://pytorch.org/docs/stable/generated/torch.no_grad.html 
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
def compute_loss(batch, policy_net, target_net):
    states, actions, rewards, next_states, dones = batch
    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * max_next_q * (1 - dones)
    return nn.functional.mse_loss(q_vals, targets)

# training
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env()
    policy_net = DQN(n_actions=5).to(device) # network that learns to predict Q-values
    target_net = DQN(n_actions=5).to(device) # target network that provides stable Q-value targets for training the policy network
    
    # load pre-trained model if exists
    checkpoints = sorted(glob.glob(str(WEIGHTS_DIR / "dqn_ep*.pt")), key=os.path.getmtime)
    total_steps = 0
    if checkpoints and PRE_TRAINED:
        print(f"Resuming from {checkpoints[-1]}")
        ckpt = torch.load(checkpoints[-1], map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            policy_net.load_state_dict(ckpt['model'])
            total_steps = ckpt['total_steps']
        else:
            policy_net.load_state_dict(ckpt)
            try:
                ep = int(os.path.basename(checkpoints[-1]).replace('dqn_ep', '').replace('.pt', ''))
                total_steps = (ep - 1) * 1000
            except ValueError:
                total_steps = 0

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR) # TODO: Experiment with different optimizers

    buffer = ReplayBuffer(BUFFER_SIZE)
    frame_stack = FrameStack(STACK_N)

    for episode in range(1, MAX_EPISODES + 1):
        # initialize the episode
        obs, _ = env.reset()
        frame_stack.reset(preprocess_without_graysscale(obs))
        state = frame_stack.get()
        episode_return = 0.0
        done = False
        eps = EPS_START

        while not done:
            # decay eps and get action from policy
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY)
            action = select_action(state, eps, policy_net, device)

            # get next state from chosen action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition in stack and replay buffer and update state
            frame_stack.push(preprocess_without_graysscale(next_obs))
            next_state = frame_stack.get()
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward
            total_steps += 1

            # updates policy network after the buffer is "warm"
            if len(buffer) >= TRAIN_START:
                batch = buffer.sample(BATCH_SIZE, device)
                loss = compute_loss(batch, policy_net, target_net)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

            # updates target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Ep {episode:4d} | Return {episode_return:8.1f} | Eps {eps:.3f} | Steps {total_steps}")

        if episode % SAVE_EVERY == 0:
            torch.save({'model': policy_net.state_dict(), 'total_steps': total_steps}, str(WEIGHTS_DIR / f"dqn_ep{episode}.pt"))
            print(f"  -> Saved dqn_ep{episode}.pt")

    torch.save(policy_net.state_dict(), str(WEIGHTS_DIR / "dqn_final.pt"))
    env.close()


if __name__ == "__main__":
    train()
