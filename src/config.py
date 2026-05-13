from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "src/runs"

# epsilon-greedy
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 150_000

# training
TRAIN_START   = 5_000
TARGET_UPDATE = 1_000
SAVE_EVERY    = 10
MAX_EPISODES  = 1_000
GAMMA         = 0.99
LR            = 1e-4
BATCH_SIZE    = 32
BUFFER_SIZE   = 300_000
STACK_N       = 4
PRE_TRAINED   = False
DOUBLE_DQN    = True
GRAYSCALE     = True

# sweep grid — variants to train and seeds per variant
VARIANTS = [(False, True), (False, False), (True, True), (True, False)]
SEEDS    = [0]


def variant_name(double_dqn: bool, grayscale: bool) -> str:
    return ("ddqn" if double_dqn else "dqn") + ("_gray" if grayscale else "_rgb")


# build a full run config dict from variant + seed; paths derived from variant_name
def make_config(double_dqn: bool, grayscale: bool, seed: int) -> dict:
    name = variant_name(double_dqn, grayscale)
    run_dir = RUNS_DIR / f"{name}_seed{seed}"
    return {
        "EPS_START":     EPS_START,
        "EPS_END":       EPS_END,
        "EPS_DECAY":     EPS_DECAY,
        "TRAIN_START":   TRAIN_START,
        "TARGET_UPDATE": TARGET_UPDATE,
        "SAVE_EVERY":    SAVE_EVERY,
        "MAX_EPISODES":  MAX_EPISODES,
        "GAMMA":         GAMMA,
        "LR":            LR,
        "BATCH_SIZE":    BATCH_SIZE,
        "BUFFER_SIZE":   BUFFER_SIZE,
        "STACK_N":       STACK_N,
        "PRE_TRAINED":   PRE_TRAINED,
        "DOUBLE_DQN":    double_dqn,
        "GRAYSCALE":     grayscale,
        "seed":          seed,
        "variant_name":  name,
        "run_dir":       str(run_dir),
        "weights_dir":   str(run_dir / "weights"),
        "plots_dir":     str(run_dir / "plots"),
        "csv_path":      str(run_dir / "episodes.csv"),
        "config_path":   str(run_dir / "config.json"),
    }
