# utils.py — Final optimized configuration for Double-Q learning traffic RL

# -----------------------------
# State discretization
# -----------------------------
MAX_BIN = 8  # queue length binning


# -----------------------------
# Traffic simulation parameters
# -----------------------------
SPAWN_PROB = 0.20          # probability of new vehicle per step
LEFT_TURN_PROB = 0.15      # left-turn ratio

EPISODE_LENGTH = 400      # steps per episode
MIN_PHASE_TIME = 10        # minimum green time before switching
SWITCH_PENALTY = 1         # discourage frequent switching


# -----------------------------
# Reinforcement Learning Settings
# -----------------------------

# High gamma = long-term planning (critical for traffic)
GAMMA = 0.99

# Epsilon-greedy exploration schedule
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_EPIS = 1500       # decays slowly → stable learning

# Learning rate decay (alpha)
ALPHA_START = 0.50
ALPHA_END = 0.05
ALPHA_DECAY_EPIS = 1500

# Optimistic initialization (forces exploration early)
OPTIMISTIC_INIT = 2.0


# -----------------------------
# Training settings
# -----------------------------
EPISODES = 4000

# Visualization (not used during training)
DRAW_SIM = False
SCREEN_W, SCREEN_H = 700, 700
FPS = 30

# Random seed
SEED = 42


# -----------------------------
# Helper Functions
# -----------------------------
def bin_count(n):
    """Convert queue length to discrete bin."""
    if n >= MAX_BIN:
        return MAX_BIN
    return int(n)
