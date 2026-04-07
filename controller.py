# controller.py — Master Controller for Multi-Agent RL (4 intersections)

import random
from collections import defaultdict
from utils import GAMMA


class MasterAgent:
    """
    Master agent controlling high-level coordination between 4 intersections.
    State: tuple of 4 local states
    Action: 0 = prioritize N-S (phases 0,2), 1 = prioritize E-W (phases 1,3)
    """

    def __init__(self, seed=None):
        self.random = random.Random(seed)

        # Single-table Q-learning for master agent
        self.Q = defaultdict(lambda: 0.0)

        # Exploration parameters
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9993     # slow decay

        # Learning rate parameters
        self.alpha = 0.5
        self.alpha_min = 0.05
        self.alpha_decay = 0.999

    # -------------------------------------------------------
    # Update epsilon ONCE per episode (called from trainer)
    # -------------------------------------------------------
    def update_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def update_alpha(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    # -------------------------------------------------------
    # Choose master action (NO epsilon update here anymore)
    # -------------------------------------------------------
    def choose_action(self, global_state):

        # Epsilon-greedy
        if self.random.random() < self.eps:
            return self.random.randint(0, 1)

        # Exploitation
        q_ns = self.Q[(global_state, 0)]
        q_ew = self.Q[(global_state, 1)]

        if q_ns > q_ew:
            return 0
        elif q_ew > q_ns:
            return 1
        else:
            return self.random.randint(0, 1)

    # -------------------------------------------------------
    # Q-learning update
    # -------------------------------------------------------
    def update(self, state, action, reward, next_state):

        old = self.Q[(state, action)]
        target = reward + GAMMA * max(
            self.Q[(next_state, 0)],
            self.Q[(next_state, 1)]
        )

        self.Q[(state, action)] = old + self.alpha * (target - old)
