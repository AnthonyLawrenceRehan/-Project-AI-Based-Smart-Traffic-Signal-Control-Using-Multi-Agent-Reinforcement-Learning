# agent.py — Final Corrected Double Q-learning Agent

import random
from collections import defaultdict
from utils import (
    GAMMA,
    EPS_START, EPS_END,
    ALPHA_START, ALPHA_END,
    OPTIMISTIC_INIT
)

class QAgent:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

        # Double Q-tables
        self.Q1 = defaultdict(lambda: OPTIMISTIC_INIT)
        self.Q2 = defaultdict(lambda: OPTIMISTIC_INIT)

        # Current epsilon & alpha
        self.eps = EPS_START
        self.alpha = ALPHA_START

        # Decay parameters (set in train.py)
        self.eps_decay = 0.997
        self.eps_min = 0.01

        self.alpha_decay = 0.999
        self.alpha_min = 0.05

    # -------------------------------
    # Update epsilon (new method)
    # -------------------------------
    def update_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # -------------------------------
    # Update alpha
    # -------------------------------
    def update_alpha(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    # -------------------------------
    # Average Q-value
    # -------------------------------
    def _avg_q(self, state, a):
        return 0.5 * (self.Q1[(state, a)] + self.Q2[(state, a)])

    # -------------------------------
    # Choose action (epsilon-greedy)
    # -------------------------------
    def choose_action(self, state, ep):

        # epsilon & alpha already updated in train.py

        # Exploration
        if self.random.random() < self.eps:
            return self.random.randint(0, 3)

        # Exploitation
        values = [self._avg_q(state, a) for a in range(4)]
        max_val = max(values)
        best = [a for a, v in enumerate(values) if v == max_val]
        return self.random.choice(best)

    # -------------------------------
    # Best action for evaluation
    # -------------------------------
    def best_action(self, state):
        values = [self._avg_q(state, a) for a in range(4)]
        max_val = max(values)
        best = [a for a, v in enumerate(values) if v == max_val]
        return self.random.choice(best)

    # -------------------------------
    # Double Q-learning update
    # -------------------------------
    def update(self, state, action, reward, next_state, ep):

        # Update alpha every step
        self.update_alpha()

        # Randomly choose which table to update
        if self.random.random() < 0.5:
            # Update Q1
            a_star = max(range(4), key=lambda a: self.Q1[(next_state, a)])
            target = reward + GAMMA * self.Q2[(next_state, a_star)]
            old = self.Q1[(state, action)]
            self.Q1[(state, action)] = old + self.alpha * (target - old)

        else:
            # Update Q2
            a_star = max(range(4), key=lambda a: self.Q2[(next_state, a)])
            target = reward + GAMMA * self.Q1[(next_state, a_star)]
            old = self.Q2[(state, action)]
            self.Q2[(state, action)] = old + self.alpha * (target - old)
