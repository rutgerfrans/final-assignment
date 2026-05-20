import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from src.model import DQN, ReplayBuffer, FrameStack, SkipFrame, preprocess_grayscale, preprocess_without_graysscale
from src.config import GAMMA, DOUBLE_DQN, GRAYSCALE, make_config


# I got the environment from Gymnasium
# https://gymnasium.farama.org/environments/box2d/car_racing/
def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    env = SkipFrame(env, skip=4)
    return env

# Epsilon-greedy action selection
# This greedy step was based on slide 8 of Lect 7 - Generalisation and Control, and the epsilon decay is a linear schedule from eps_start to eps_end over eps_decay steps.
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
# The Double-DQN line is based on slide 13 from Lect 7 - Generalisation and Control, where the best action is selected using the policy network but the target Q-value is obtained from the target network.
def compute_loss(batch, policy_net, target_net, double_dqn: bool):
    states, actions, rewards, next_states, dones = batch
    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            best_actions = policy_net(next_states).argmax(1, keepdim=True)
            max_next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
        else:
            max_next_q = target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * max_next_q * (1 - dones)
    return nn.functional.mse_loss(q_vals, targets)

# helper function for plotting
def save_return_plot(episode_numbers, returns, double_dqn: bool, grayscale: bool, path):
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
    colortype = "grayscale" if grayscale else "RGB"
    ax.set_title(f"{variant} CarRacing-v3 ({colortype}) — returns up to episode {episode_numbers[-1]}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close("all")


# training
def train(cfg: dict):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grayscale = cfg["GRAYSCALE"]
    double_dqn = cfg["DOUBLE_DQN"]
    stack_n = cfg["STACK_N"]
    preprocess_fn = preprocess_grayscale if grayscale else preprocess_without_graysscale
    frame_shape = (1, 84, 96) if grayscale else (3, 84, 96)
    in_channels = stack_n * (1 if grayscale else 3)

    weights_dir = Path(cfg["weights_dir"])
    plots_dir = Path(cfg["plots_dir"])
    csv_path = Path(cfg["csv_path"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg["config_path"], "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"\n{'='*52}\n"
        f"  Variant     : {cfg['variant_name']}  (seed {seed})\n"
        f"  Device      : {device}\n"
        f"  Input       : {'Grayscale (1ch)' if grayscale else 'RGB (3ch)'}\n"
        f"  Episodes    : {cfg['MAX_EPISODES']}  |  Batch: {cfg['BATCH_SIZE']}  |  Buffer: {cfg['BUFFER_SIZE']:,}\n"
        f"  LR          : {cfg['LR']}  |  Gamma: {cfg['GAMMA']}  |  Stack: {stack_n}\n"
        f"  Eps         : {cfg['EPS_START']} -> {cfg['EPS_END']} over {cfg['EPS_DECAY']:,} steps\n"
        f"  Target upd  : every {cfg['TARGET_UPDATE']:,} steps\n"
        f"{'='*52}\n"
    )

    env = make_env()
    env.action_space.seed(seed)

    policy_net = DQN(n_actions=5, in_channels=in_channels).to(device) # network that learns to predict Q-values
    target_net = DQN(n_actions=5, in_channels=in_channels).to(device) # target network that provides stable Q-value targets for training the policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["LR"])
    buffer = ReplayBuffer(cfg["BUFFER_SIZE"], stack_n=stack_n, frame_shape=frame_shape)
    frame_stack = FrameStack(stack_n)

    eps_start = cfg["EPS_START"]
    eps_end = cfg["EPS_END"]
    eps_decay = cfg["EPS_DECAY"]
    train_start = cfg["TRAIN_START"]
    target_update = cfg["TARGET_UPDATE"]
    save_every = cfg["SAVE_EVERY"]
    max_episodes = cfg["MAX_EPISODES"]
    batch_size = cfg["BATCH_SIZE"]

    total_steps = 0
    episode_numbers = []
    all_returns = []
    start_time = time.time()
    first_reset = True

    # warm start from the most recent checkpoint in this run's weights dir
    if cfg["PRE_TRAINED"] and weights_dir.exists():
        ckpts = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            print(f"Resuming from {ckpts[-1]}")
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                policy_net.load_state_dict(ckpt['model'])
                total_steps = ckpt.get('total_steps', 0)
            else:
                policy_net.load_state_dict(ckpt)
            target_net.load_state_dict(policy_net.state_dict())

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["episode", "return", "epsilon_end", "total_steps",
                     "wall_clock_seconds", "loss_mean", "loss_count"])

    for episode in range(1, max_episodes + 1):
        # initialize the episode
        if first_reset:
            obs, _ = env.reset(seed=seed)
            first_reset = False
        else:
            obs, _ = env.reset()
        init_frame = preprocess_fn(obs)
        frame_stack.reset(init_frame)
        buffer.reset_episode(init_frame)
        state = frame_stack.get()
        episode_return = 0.0
        done = False
        eps = eps_start
        loss_sum = 0.0
        loss_count = 0

        # most of this part is based on the pseudcode from slide 8 of Lect 7 - Generalisation and Control
        while not done:
            # decay eps, get action from policy, and get next state from this chosen action
            
            eps = max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay)
            action = select_action(state, eps, policy_net, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition in stack and replay buffer and update state
            new_frame = preprocess_fn(next_obs)
            frame_stack.push(new_frame)
            next_state = frame_stack.get()
            buffer.push(new_frame, action, reward, float(done))
            state = next_state
            episode_return += reward
            total_steps += 1

            # updates policy network after the buffer is warm
            if len(buffer) >= train_start:
                batch = buffer.sample(batch_size, device)
                loss = compute_loss(batch, policy_net, target_net, double_dqn)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()
                loss_sum += loss.item()
                loss_count += 1

            # updates target network
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_numbers.append(episode)
        all_returns.append(episode_return)
        wall = time.time() - start_time
        loss_mean = (loss_sum / loss_count) if loss_count > 0 else 0.0
        print(f"Ep {episode:4d} | Return {episode_return:8.1f} | Eps {eps:.3f} | Steps {total_steps}")
        writer.writerow([episode, episode_return, eps, total_steps, wall, loss_mean, loss_count])
        csv_file.flush()

        # sparse milestone checkpoints
        if episode in (500, 1000):
            torch.save({'model': policy_net.state_dict(), 'total_steps': total_steps},
                       str(weights_dir / f"episode_{episode}.pt"))

        # progress plot — overwrite a single file so partial runs are monitorable
        if episode % save_every == 0:
            save_return_plot(episode_numbers, all_returns, double_dqn, grayscale,
                             plots_dir / "returns_latest.png")

    save_return_plot(episode_numbers, all_returns, double_dqn, grayscale,
                     plots_dir / "returns_final.png")
    torch.save({'model': policy_net.state_dict(), 'total_steps': total_steps},
               str(weights_dir / "final.pt"))
    csv_file.close()
    env.close()


if __name__ == "__main__":
    train(make_config(DOUBLE_DQN, GRAYSCALE, seed=0))
