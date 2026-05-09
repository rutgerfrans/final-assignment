import argparse
import numpy as np
import torch
import gymnasium as gym
from src.model import DQN, FrameStack, SkipFrame, preprocess_without_graysscale
from src.config import STACK_N

# run one episode and return the total reward
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

# evaluate a trained model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if args.render else None
    env = SkipFrame(gym.make("CarRacing-v3", continuous=False, render_mode=render_mode), skip=4)

    net = DQN(n_actions=5).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    net.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    net.eval()

    returns = []
    for i in range(args.episodes):
        r = run_episode(env, net, device)
        returns.append(r)
        print(f"Episode {i + 1}: {r:.1f}")

    print(f"\nMean return over {args.episodes} episodes: {np.mean(returns):.1f}")
    env.close()


if __name__ == "__main__":
    main()
