import glob
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from src.model import DQN, ReplayBuffer, FrameStack, SkipFrame, preprocess_without_graysscale
from src.config import (ROOT, WEIGHTS_DIR, EPS_START, EPS_END, EPS_DECAY, TRAIN_START,
                        TARGET_UPDATE, SAVE_EVERY, MAX_EPISODES, GAMMA,
                        LR, BATCH_SIZE, BUFFER_SIZE, STACK_N, PRE_TRAINED, DOUBLE_DQN)

PLOTS_DIR = WEIGHTS_DIR.parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# I got the environment from Gymnasium
# https://gymnasium.farama.org/environments/box2d/car_racing/
def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    env = SkipFrame(env, skip=4)
    return env

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
def compute_loss(batch, policy_net, target_net, double_dqn: bool):
    states, actions, rewards, next_states, dones = batch
    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            # use policy_net to select the action, target_net to evaluate it
            best_actions = policy_net(next_states).argmax(1, keepdim=True)
            max_next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
        else:
            max_next_q = target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * max_next_q * (1 - dones)
    return nn.functional.mse_loss(q_vals, targets)

# helper function for plotting
def save_return_plot(episode_numbers, returns, checkpoint_episode, double_dqn: bool):
    window = max(1, len(returns) // 10)
    rolling = None
    if len(returns) >= window:
        rolling = [
            sum(returns[i:i + window]) / window
            for i in range(len(returns) - window + 1)
        ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode_numbers, returns, linewidth=0.8, alpha=0.4, color="steelblue", label="return")
    if rolling:
        ax.plot(episode_numbers[window - 1:], rolling, linewidth=2.0, color="steelblue",
                label=f"rolling avg (w={window})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    variant = "Double DQN" if double_dqn else "DQN"
    ax.set_title(f"{variant} CarRacing-v3 — returns up to episode {checkpoint_episode}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    variant_tag = "ddqn" if double_dqn else "dqn"
    path = PLOTS_DIR / f"returns_{variant_tag}_ep{checkpoint_episode:04d}.png"
    fig.savefig(path, dpi=120)
    plt.close("all")
    print(f"  -> Plot saved to {path.relative_to(ROOT)}")


# training
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Double DQN: {DOUBLE_DQN}")

    env = make_env()
    policy_net = DQN(n_actions=5).to(device) # network that learns to predict Q-values
    target_net = DQN(n_actions=5).to(device) # target network that provides stable Q-value targets for training the policy network

    checkpoints = sorted(glob.glob(str(WEIGHTS_DIR / "dqn_ep*.pt")), key=os.path.getmtime)
    total_steps = 0
    start_episode = 1
    episode_numbers = []
    all_returns = []

    if checkpoints and PRE_TRAINED:
        print(f"Resuming from {checkpoints[-1]}")
        ckpt = torch.load(checkpoints[-1], map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            policy_net.load_state_dict(ckpt['model'])
            total_steps = ckpt['total_steps']
        else:
            policy_net.load_state_dict(ckpt)
            try:
                ep = int(os.path.basename(checkpoints[-1]).replace('dqn_ep', '').replace('.pt', ''))
                total_steps = (ep - 1) * 1000
                start_episode = ep + 1
            except ValueError:
                pass

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR) # TODO: Experiment with different optimizers
    buffer = ReplayBuffer(BUFFER_SIZE, stack_n=STACK_N)
    frame_stack = FrameStack(STACK_N)

    for episode in range(start_episode, MAX_EPISODES + 1):
        # initialize the episode
        obs, _ = env.reset()
        init_frame = preprocess_without_graysscale(obs)
        frame_stack.reset(init_frame)
        buffer.reset_episode(init_frame)
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
            new_frame = preprocess_without_graysscale(next_obs)
            frame_stack.push(new_frame)
            next_state = frame_stack.get()
            buffer.push(new_frame, action, reward, float(done))
            state = next_state
            episode_return += reward
            total_steps += 1

            # updates policy network after the buffer is "warm"
            if len(buffer) >= TRAIN_START:
                batch = buffer.sample(BATCH_SIZE, device)
                loss = compute_loss(batch, policy_net, target_net, DOUBLE_DQN)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

            # updates target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_numbers.append(episode)
        all_returns.append(episode_return)
        print(f"Ep {episode:4d} | Return {episode_return:8.1f} | Eps {eps:.3f} | Steps {total_steps}")

        if episode % SAVE_EVERY == 0:
            torch.save({'model': policy_net.state_dict(), 'total_steps': total_steps},
                       str(WEIGHTS_DIR / f"dqn_ep{episode}.pt"))
            print(f"  -> Saved dqn_ep{episode}.pt")
            save_return_plot(episode_numbers, all_returns, episode, DOUBLE_DQN)

    torch.save(policy_net.state_dict(), str(WEIGHTS_DIR / "dqn_final.pt"))
    env.close()


if __name__ == "__main__":
    train()
