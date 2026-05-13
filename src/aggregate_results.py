"""Aggregate the 12 sweep CSVs into mean ± std plots and a summary table."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.config import RUNS_DIR


VARIANTS = ["dqn_gray", "dqn_rgb", "ddqn_gray", "ddqn_rgb"]
SEEDS    = [0]
AGG_DIR  = RUNS_DIR / "aggregate"
ROLL_W   = 100
DUR_W    = 50


def variant_title(v):
    algo = "Double DQN" if v.startswith("ddqn") else "DQN"
    inp  = "grayscale" if v.endswith("gray") else "RGB"
    return f"{algo} ({inp})"


def load_run(variant, seed):
    path = RUNS_DIR / f"{variant}_seed{seed}" / "episodes.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    returns = np.array([float(r["return"]) for r in rows], dtype=np.float64)
    wall    = np.array([float(r["wall_clock_seconds"]) for r in rows], dtype=np.float64)
    return returns, wall


def rolling_mean(x, w):
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    # load everything first so window sizes can clamp to actual data length
    loaded = {v: [load_run(v, s) for s in SEEDS] for v in VARIANTS}
    n = min(len(r) for v in VARIANTS for r, _ in loaded[v])
    roll_w  = min(ROLL_W, n)
    dur_w   = min(DUR_W, n)
    final_n = min(100, n)

    summary_rows = []
    per_variant_roll    = {}   # (mean, std)
    per_variant_dur     = {}   # mean per-episode duration over training
    per_variant_totals  = {}   # total runtime per seed

    for v in VARIANTS:
        returns_arr = np.stack([r[:n] for r, _ in loaded[v]])
        # first episode's duration is wall[0]; subsequent are diffs
        dur_arr     = np.stack([np.diff(w[:n], prepend=0.0) for _, w in loaded[v]])
        totals      = np.array([w[n - 1] for _, w in loaded[v]])

        roll_per_seed = np.stack([rolling_mean(returns_arr[i], roll_w) for i in range(len(returns_arr))])
        roll_mean = roll_per_seed.mean(axis=0)
        roll_std  = roll_per_seed.std(axis=0)
        per_variant_roll[v] = (roll_mean, roll_std)

        dur_roll = np.stack([rolling_mean(dur_arr[i], dur_w) for i in range(len(dur_arr))])
        per_variant_dur[v] = dur_roll.mean(axis=0)
        per_variant_totals[v] = totals

        final_per_seed = returns_arr[:, -final_n:].mean(axis=1)
        best_per_seed  = roll_per_seed.max(axis=1)
        ep_dur_per_seed = dur_arr.mean(axis=1)
        summary_rows.append({
            "variant": v,
            "final_return_mean":      float(final_per_seed.mean()),
            "final_return_std":       float(final_per_seed.std()),
            "best_rolling_return_mean": float(best_per_seed.mean()),
            "best_rolling_return_std":  float(best_per_seed.std()),
            "total_runtime_seconds_mean": float(totals.mean()),
            "total_runtime_seconds_std":  float(totals.std()),
            "mean_episode_duration_seconds_mean": float(ep_dur_per_seed.mean()),
            "mean_episode_duration_seconds_std":  float(ep_dur_per_seed.std()),
        })

        x = np.arange(roll_w, roll_w + len(roll_mean))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, roll_mean, linewidth=2.0, color="steelblue", label=f"mean (w={roll_w})")
        ax.fill_between(x, roll_mean - roll_std, roll_mean + roll_std,
                        color="steelblue", alpha=0.2, label="±1 std")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rolling return")
        ax.set_title(f"{variant_title(v)} — return across {len(SEEDS)} seeds")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(AGG_DIR / f"{v}_returns.png", dpi=120)
        plt.close(fig)

    # combined plot — mean lines only for legibility
    fig, ax = plt.subplots(figsize=(10, 4))
    for v in VARIANTS:
        roll_mean, _ = per_variant_roll[v]
        x = np.arange(roll_w, roll_w + len(roll_mean))
        ax.plot(x, roll_mean, linewidth=1.8, label=variant_title(v))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling return (w={roll_w})")
    ax.set_title("CarRacing-v3 — mean rolling return across seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "all_variants_returns.png", dpi=120)
    plt.close(fig)

    # per-episode runtime over training
    fig, ax = plt.subplots(figsize=(10, 4))
    for v in VARIANTS:
        dur = per_variant_dur[v]
        x = np.arange(dur_w, dur_w + len(dur))
        ax.plot(x, dur, linewidth=1.5, label=variant_title(v))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Per-episode duration (s)")
    ax.set_title(f"Per-episode runtime across seeds (rolling w={dur_w})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "runtime_per_episode.png", dpi=120)
    plt.close(fig)

    # total runtime bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    means_min = [per_variant_totals[v].mean() / 60 for v in VARIANTS]
    stds_min  = [per_variant_totals[v].std()  / 60 for v in VARIANTS]
    labels    = [variant_title(v) for v in VARIANTS]
    ax.bar(labels, means_min, yerr=stds_min, capsize=6, color="steelblue", alpha=0.85)
    ax.set_ylabel("Total training time (min)")
    ax.set_title("Total runtime per variant (mean ± std across seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "runtime_total.png", dpi=120)
    plt.close(fig)

    cols = ["variant",
            "final_return_mean", "final_return_std",
            "best_rolling_return_mean", "best_rolling_return_std",
            "total_runtime_seconds_mean", "total_runtime_seconds_std",
            "mean_episode_duration_seconds_mean", "mean_episode_duration_seconds_std"]
    with open(AGG_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"Wrote aggregate outputs to {AGG_DIR}")


if __name__ == "__main__":
    main()
