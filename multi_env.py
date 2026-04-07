# multi_env.py — Environment for 4-intersection + master RL

from env import TrafficEnvMixed
import random

class MultiIntersectionEnv4:
    """
    Environment with 4 independent intersections + master coordination.
    Each intersection uses TrafficEnvMixed logic but shares:
      - global priority from master
      - aggregated reward
    """

    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.intersections = [
            TrafficEnvMixed(seed=seed + i) for i in range(4)
        ]

        # Master mode: 0 = prioritize NS, 1 = prioritize EW
        self.global_priority = 0

        self.t = 0
        self.done = False

    def reset(self):
        self.t = 0
        self.done = False
        self.global_priority = 0  # default

        # Reset all 4 intersections
        states = []
        for inter in self.intersections:
            states.append(inter.reset())

        # Global state = tuple of 4 states
        return tuple(states)

    def step(self):
        self.t += 1

        total_wait = 0
        total_pass = 0
        total_block = 0
        new_states = []

        # Apply master priority rule
        # 0 = NS priority → phases {0,2}
        # 1 = EW priority → phases {1,3}

        for i, inter in enumerate(self.intersections):

            if self.global_priority == 0:
                # Force NS-heavy behavior
                if inter.phase in (1, 3):  # EW phases
                    inter.phase = 0
            else:
                # Force EW-heavy behavior
                if inter.phase in (0, 2):  # NS phases
                    inter.phase = 1

            # Take one environment step
            next_state, reward, done_flag, info = inter.step(inter.phase)

            new_states.append(next_state)
            total_wait += info["total_waiting"]
            total_pass += info["passed"]
            total_block += info["blocked"]

        # Episode ends when any environment ends OR after 300 steps
        done = self.t >= 300

        # Global reward for master
        global_reward = (
            3.0 * total_pass
            - 0.1 * total_wait
            - 2.0 * total_block
        )

        return tuple(new_states), global_reward, done, {
            "total_wait": total_wait,
            "total_pass": total_pass,
            "total_block": total_block
        }
