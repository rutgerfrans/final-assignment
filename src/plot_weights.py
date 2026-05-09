import argparse
import re
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import STACK_N
from src.model import DQN, FrameStack, preprocess_without_graysscale

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights/dqn_rgb_carracing_weights"


def run_episode(env, net, device):
    obs, _ = env.reset()
    frame_stack = FrameStack(STACK_N)
    frame_stack.reset(preprocess_without_graysscale(obs))
    total_return = 0.0
    done = False
    while not done:
        state = frame_stack.get()
        with torch.no_grad():
            s = torch.tensor(state[None], dtype=torch.float32, device=device)
            action = net(s).argmax(dim=1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame_stack.push(preprocess_without_graysscale(obs))
        total_return += reward
    return total_return


def episode_number(path: Path) -> int:
    match = re.search(r"ep(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1, help="episodes to average per checkpoint (default: 1)")
    parser.add_argument("--output", type=str, default=None, help="save figure to this path instead of displaying")
    args = parser.parse_args()

    checkpoints = sorted(WEIGHTS_DIR.glob("*.pt"), key=episode_number)
    if not checkpoints:
        print(f"No .pt files found in {WEIGHTS_DIR}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints — running {args.episodes} episode(s) each.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)

    episodes_axis = []
    mean_returns = []
    std_returns = []

    for i, ckpt in enumerate(checkpoints):
        ep_num = episode_number(ckpt)
        net = DQN(n_actions=5).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        net.eval()

        returns = [run_episode(env, net, device) for _ in range(args.episodes)]
        mean = float(np.mean(returns))
        std = float(np.std(returns))

        episodes_axis.append(ep_num)
        mean_returns.append(mean)
        std_returns.append(std)

        print(f"[{i+1}/{len(checkpoints)}] ep{ep_num:4d}  return={mean:.1f}" +
              (f" ± {std:.1f}" if args.episodes > 1 else ""))

    env.close()

    mean_returns = np.array(mean_returns)
    std_returns = np.array(std_returns)

    window = max(1, len(mean_returns) // 10)  # ~10% of checkpoints
    rolling = np.convolve(mean_returns, np.ones(window) / window, mode="valid")
    rolling_x = episodes_axis[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes_axis, mean_returns, linewidth=0.8, color="steelblue", alpha=0.35, label="per-checkpoint return")
    if args.episodes > 1:
        ax.fill_between(
            episodes_axis,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.15,
            color="steelblue",
        )
    ax.plot(rolling_x, rolling, linewidth=2.0, color="steelblue", label=f"rolling avg (window={window})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("Episode return")
    ax.set_title("DQN RGB CarRacing-v3 — return per checkpoint")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Figure saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
