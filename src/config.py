from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# epsilon-greedy
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 200_000

# training
TRAIN_START   = 5_000
TARGET_UPDATE = 1_000
SAVE_EVERY    = 10
MAX_EPISODES  = 1_000
GAMMA         = 0.99
LR            = 1e-4
BATCH_SIZE    = 32
BUFFER_SIZE   = 10_000
STACK_N       = 4
PRE_TRAINED   = True
