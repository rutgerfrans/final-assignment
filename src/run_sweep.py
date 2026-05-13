"""Run all 12 (algo, input, seed) combinations sequentially.

Skips runs whose episodes.csv already shows MAX_EPISODES completed episodes,
so the script is restartable.
"""
import time
from pathlib import Path

from src.config import MAX_EPISODES, SEEDS, VARIANTS, make_config
from src.train import train


# True iff the csv has at least header + expected data rows
def csv_complete(csv_path, expected):
    if not csv_path.exists():
        return False
    with open(csv_path) as f:
        return sum(1 for _ in f) >= expected + 1


def main():
    runs = [(d, g, s) for d, g in VARIANTS for s in SEEDS]
    durations = []

    for i, (double_dqn, grayscale, seed) in enumerate(runs, start=1):
        cfg = make_config(double_dqn, grayscale, seed)
        csv_path = Path(cfg["csv_path"])
        key = f"{cfg['variant_name']}_seed{seed}"

        if csv_complete(csv_path, MAX_EPISODES):
            print(f"\n[{i}/{len(runs)}] {key}: already complete, skipping")
            continue

        remaining = len(runs) - i + 1
        eta_str = "n/a"
        if durations:
            avg = sum(durations) / len(durations)
            eta_str = f"~{avg * remaining / 60:.0f} min"

        print(
            f"\n{'='*52}\n"
            f"  Sweep run   : {i}/{len(runs)}  ({key})\n"
            f"  Algorithm   : {'Double DQN' if double_dqn else 'DQN'}\n"
            f"  Input       : {'Grayscale' if grayscale else 'RGB'}\n"
            f"  Seed        : {seed}\n"
            f"  ETA remain  : {eta_str}\n"
            f"{'='*52}\n"
        )

        start = time.time()
        train(cfg)
        elapsed = time.time() - start
        durations.append(elapsed)
        print(f"  -> {key} done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
