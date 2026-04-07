# env_grid4.py — 4-Intersection Traffic Environment for Multi-Agent RL
# Uses your existing TrafficEnvMixed logic for each intersection.

from collections import deque
import random
from utils import SPAWN_PROB, LEFT_TURN_PROB, MIN_PHASE_TIME, EPISODE_LENGTH, bin_count


class Intersection:
    """Single intersection with the same logic as TrafficEnvMixed."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.queues = {d: deque() for d in ['N','S','E','W']}
        self.free_left = {'N': True, 'S': False, 'E': True, 'W': False}
        self.phase = 0
        self.time_in_phase = 0
        self.t = 0
        self.total_wait_time = 0.0

    def reset(self):
        for q in self.queues.values():
            q.clear()
        self.phase = 0
        self.time_in_phase = 0
        self.t = 0
        self.total_wait_time = 0.0
        return self.get_state()

    def spawn_vehicles(self):
        for d in ['N','S','E','W']:
            if self.rng.random() < SPAWN_PROB:
                vtype = 'L' if self.rng.random() < LEFT_TURN_PROB else 'S'
                self.queues[d].append({'type': vtype, 'wait': 0})

    def free_left_turns(self):
        passed = 0
        for d in self.queues:
            if self.free_left[d]:
                if self.queues[d] and self.queues[d][0]['type'] == 'L':
                    self.queues[d].popleft()
                    passed += 1
        return passed

    def signal_release(self):
        passed = 0
        # Phases identical to your TrafficEnvMixed
        if self.phase == 0:
            allowed = {'N':'S','S':'S'}
        elif self.phase == 1:
            allowed = {'E':'S','W':'S'}
        elif self.phase == 2:
            allowed = {'N':'L','S':'L'}
        else:
            allowed = {'E':'L','W':'L'}

        for d, need in allowed.items():
            if self.queues[d] and self.queues[d][0]['type'] == need:
                self.queues[d].popleft()
                passed += 1
        return passed

    def age_queues(self):
        for q in self.queues.values():
            for veh in q:
                veh['wait'] += 1.0
                self.total_wait_time += 1.0

    def step_local(self, action):
        """Step for one intersection."""
        wait_before = self.total_wait_time

        # Phase switching
        if action != self.phase and self.time_in_phase >= MIN_PHASE_TIME:
            self.phase = action
            self.time_in_phase = 0

        self.spawn_vehicles()
        free_pass = self.free_left_turns()
        signal_pass = self.signal_release()
        passed = free_pass + signal_pass

        self.age_queues()

        wait_delta = self.total_wait_time - wait_before

        self.time_in_phase += 1
        self.t += 1

        return passed, wait_delta, self.get_state()

    def get_state(self):
        bins = []
        front = []
        for d in ['N','S','E','W']:
            q = self.queues[d]
            bins.append(bin_count(len(q)))
            if not q:
                front.append(0)
            else:
                front.append(1 if q[0]['type'] == 'S' else 2)
        return tuple(bins + front + [self.phase])


# ---------------------------------------------------------------
#                   GRID ENV (4 Intersections)
# ---------------------------------------------------------------

class GridEnv4:
    """
    4-intersection grid:
       0 ─ 1
       |   |
       2 ─ 3
    """

    # Mapping: vehicles from (node, direction) → (next node, direction)
    TRANSFER = {
        (0,'E'): (1,'W'), (0,'S'): (2,'N'),
        (1,'W'): (0,'E'), (1,'S'): (3,'N'),
        (2,'N'): (0,'S'), (2,'E'): (3,'W'),
        (3,'N'): (1,'S'), (3,'W'): (2,'E')
    }

    def __init__(self, seed=0):
        rnd = random.Random(seed)
        self.inters = [Intersection(seed=rnd.randint(0,99999)) for _ in range(4)]
        self.t = 0

    def reset(self):
        self.t = 0
        return [inter.reset() for inter in self.inters]

    def step(self, actions):
        outputs = []
        passed_dirs = [ {'N':0,'S':0,'E':0,'W':0} for _ in range(4) ]

        # Step each intersection
        for i, act in enumerate(actions):
            passed, wait_delta, state = self.inters[i].step_local(act)

            # Count per-direction passed
            if act == 0:
                if 'N' in self.inters[i].queues: passed_dirs[i]['N'] += 1
                if 'S' in self.inters[i].queues: passed_dirs[i]['S'] += 1
            if act == 1:
                if 'E' in self.inters[i].queues: passed_dirs[i]['E'] += 1
                if 'W' in self.inters[i].queues: passed_dirs[i]['W'] += 1

            outputs.append({
                'passed': passed,
                'wait_delta': wait_delta,
                'state': state
            })

        # Handle transfers between intersections
        for i in range(4):
            for d, count in passed_dirs[i].items():
                if count > 0:
                    if (i, d) in self.TRANSFER:
                        to_idx, to_dir = self.TRANSFER[(i,d)]
                        for _ in range(count):
                            self.inters[to_idx].queues[to_dir].append({'type':'S','wait':0})

        self.t += 1
        done = self.t >= EPISODE_LENGTH
        return outputs, done
