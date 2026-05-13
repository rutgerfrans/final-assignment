# Multi-seed sweep — DQN / Double-DQN × grayscale / RGB

This document covers the 12-run sweep that compares DQN against Double-DQN under
grayscale and RGB inputs on CarRacing-v3, with 3 seeds per configuration.

## Launching the sweep

```bash
python3 -m src.run_sweep
```

The orchestrator iterates the cross product of:

- algorithm: DQN, Double-DQN
- input:     grayscale, RGB
- seed:      0, 1, 2

Each run trains for `MAX_EPISODES = 1000` (set in `src/config.py`) and writes to
`runs/<variant>_seed<N>/`. The script is restartable: any run whose
`episodes.csv` already contains 1000 rows is skipped.

Hyperparameters are shared across runs — only `DOUBLE_DQN`, `GRAYSCALE`, and
`seed` change. Each run dumps the full config it used to `config.json` for
reproducibility.

A single training run (no sweep) still works the way it did before:

```bash
python3 -m src.train
```

This uses `DOUBLE_DQN` and `GRAYSCALE` from `src/config.py` and seed 0.

## Expected runtime

Roughly 30–60 minutes per run on a single CUDA GPU, depending on grayscale vs.
RGB (RGB is heavier per step). 12 runs sequential — plan on ~6–10 hours total.
The orchestrator prints a rolling ETA based on completed runs.

## Per-run outputs

```
runs/<variant>_seed<N>/
  config.json            full run config
  episodes.csv           one row per episode
  plots/returns_latest.png   overwritten every SAVE_EVERY episodes
  plots/returns_final.png    written once at end of training
  weights/final.pt
  weights/best.pt            highest 100-ep rolling-mean return so far
  weights/episode_500.pt     milestone snapshot
  weights/episode_1000.pt
```

`episodes.csv` columns: `episode, return, epsilon_end, total_steps,
wall_clock_seconds, loss_mean, loss_count`. The file is flushed after every
episode, so a killed run is recoverable up to its last completed episode.

## Aggregating results

```bash
python3 -m src.aggregate_results
```

Writes to `runs/aggregate/`:

- `<variant>_returns.png` — one per variant. Rolling-mean return (window=100)
  averaged across 3 seeds, with a ±1 std band. Use these to compare seed
  variance within a single variant.
- `all_variants_returns.png` — the headline figure. Mean rolling-return lines
  for all 4 variants overlaid, no bands.
- `runtime_per_episode.png` — per-episode wall-clock duration across training
  (rolling w=50, averaged across seeds). Shows whether RGB or DDQN costs more
  per step in practice. Early episodes are faster because gradient updates
  only start after `TRAIN_START` steps; the rolling window absorbs this.
- `runtime_total.png` — bar chart of total training time per variant (minutes,
  mean ± std across seeds).
- `summary.csv` — one row per variant with columns:
  `final_return_mean/std` (mean over the last 100 episodes),
  `best_rolling_return_mean/std` (peak of the 100-ep rolling return),
  `total_runtime_seconds_mean/std`,
  `mean_episode_duration_seconds_mean/std`.

The summary CSV is the table you cite in the report; the combined plot is the
headline figure.

## Reproducibility note

Each run seeds `random`, `numpy`, `torch`, `torch.cuda`, the environment's
`reset(seed=…)` (on the first reset only), and `env.action_space`. Two
back-to-back invocations of the same seed should produce nearly identical
`episodes.csv` files. Minor numerical drift can still occur because CUDA
convolutions are not bit-exact across runs; if you need strict determinism,
swap the device to CPU or set `torch.use_deterministic_algorithms(True)` —
both will significantly slow training.
