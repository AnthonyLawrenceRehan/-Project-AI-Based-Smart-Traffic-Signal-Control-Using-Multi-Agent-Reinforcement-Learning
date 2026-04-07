# env.py — Final Optimized Mixed-Lane Traffic Environment (India Free-Left)
from collections import deque
import random
from utils import (
    SPAWN_PROB, LEFT_TURN_PROB, MIN_PHASE_TIME,
    EPISODE_LENGTH, bin_count
)

class TrafficEnvMixed:
    """
    Improved RL Environment with:
    - prediction-enhanced state (bins + predicted bins)
    - safety-filter for phase switching
    - multi-objective reward (throughput, queue penalty, wait penalty)
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.queues = {d: deque() for d in ['N','S','E','W']}
        self.free_left = {'N': True, 'S': False, 'E': True, 'W': False}

        self.phase = 0
        self.time_in_phase = 0
        self.t = 0

        self.total_wait_time = 0.0
        self.passed_vehicles = 0

        # NEW — store previous bins for prediction
        self.prev_bins = [0, 0, 0, 0]

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------
    def reset(self):
        for d in self.queues:
            self.queues[d].clear()
        self.phase = 0
        self.time_in_phase = 0
        self.t = 0
        self.total_wait_time = 0.0
        self.passed_vehicles = 0
        self.prev_bins = [0, 0, 0, 0]
        return self._get_state()

    # ------------------------------------------------------
    # STEP FUNCTION (IMPROVED)
    # ------------------------------------------------------
    def step(self, action):
        wait_before = self.total_wait_time

        # ================================
        # SAFETY FILTER FOR PHASE SWITCH
        # ================================
        switched = False
        if action != self.phase:

            # queue-based safety rule
            queue_sizes = [len(self.queues[d]) for d in ['N','S','E','W']]
            if abs(max(queue_sizes) - min(queue_sizes)) < 2:
                action = self.phase  # avoid switching when queues similar

            # only switch if phase duration minimum is achieved
            if action != self.phase and self.time_in_phase >= MIN_PHASE_TIME:
                self.phase = action
                self.time_in_phase = 0
                switched = True

        # Spawn vehicles
        self._spawn()

        # Free-left logic
        free_left_passed = self._apply_free_left()

        # Vehicles passed by signal
        signal_passed = self._signal_release()
        passed = free_left_passed + signal_passed

        # Age queues
        self._age_queues()

        # waiting added this step
        wait_delta = self.total_wait_time - wait_before

        # Compute blocked vehicles
        blocked = self._compute_blocked(self.phase)

        # Total queue length
        queue_length = sum(len(q) for q in self.queues.values())

        # ================================
        # MULTI-OBJECTIVE REWARD
        # ================================
        reward = (
            5.0 * passed            # encourage clearing traffic
            - 0.1 * wait_delta      # mild wait penalty
            - 0.5 * queue_length    # large queue penalty
            - 2.0 * blocked         # avoid conflicts
        )

        if switched:
            reward -= 1.0           # small cost for switching

        self.time_in_phase += 1
        self.t += 1
        done = (self.t >= EPISODE_LENGTH)

        return self._get_state(), reward, done, {
            'passed': passed,
            'wait_delta': wait_delta,
            'blocked': blocked,
            'total_waiting': queue_length
        }

    # ------------------------------------------------------
    # INTERNAL SIMULATION FUNCTIONS
    # ------------------------------------------------------
    def _spawn(self):
        for d in ['N','S','E','W']:
            if self.rng.random() < SPAWN_PROB:
                vtype = 'L' if self.rng.random() < LEFT_TURN_PROB else 'S'
                self.queues[d].append({'type': vtype, 'wait': 0})

    def _apply_free_left(self):
        passed = 0
        for d in ['N','S','E','W']:
            if self.free_left[d]:
                if self.queues[d] and self.queues[d][0]['type'] == 'L':
                    self.queues[d].popleft()
                    passed += 1
        return passed

    def _signal_release(self):
        passed = 0
        if self.phase == 0:
            allowed = {'N': 'S', 'S': 'S'}
        elif self.phase == 1:
            allowed = {'E': 'S', 'W': 'S'}
        elif self.phase == 2:
            allowed = {'N': 'L', 'S': 'L'}
        else:
            allowed = {'E': 'L', 'W': 'L'}

        for d, need in allowed.items():
            if self.queues[d] and self.queues[d][0]['type'] == need:
                self.queues[d].popleft()
                passed += 1

        return passed

    def _compute_blocked(self, phase):
        if phase == 0:
            dirs, need = ['N','S'], 'S'
        elif phase == 1:
            dirs, need = ['E','W'], 'S'
        elif phase == 2:
            dirs, need = ['N','S'], 'L'
        else:
            dirs, need = ['E','W'], 'L'

        blocked = 0
        for d in dirs:
            if self.queues[d] and self.queues[d][0]['type'] != need:
                blocked += 1
        return blocked

    def _age_queues(self):
        for q in self.queues.values():
            for veh in q:
                veh['wait'] += 1.0
                self.total_wait_time += 1.0

    # ------------------------------------------------------
    # FINAL STATE — WITH TRAFFIC PREDICTION
    # ------------------------------------------------------
    def _get_state(self):
        bins = []
        front = []

        for d in ['N','S','E','W']:
            q = self.queues[d]

            # bin count
            bins.append(bin_count(len(q)))

            # front vehicle type
            if not q:
                front.append(0)
            else:
                front.append(1 if q[0]['type'] == 'S' else 2)

        # traffic prediction using moving average
        predicted = [
            0.7 * bins[i] + 0.3 * self.prev_bins[i]
            for i in range(4)
        ]

        # update for next step
        self.prev_bins = bins.copy()

        # state = bins + front + predicted + phase
        return tuple(bins + front + predicted + [self.phase])
