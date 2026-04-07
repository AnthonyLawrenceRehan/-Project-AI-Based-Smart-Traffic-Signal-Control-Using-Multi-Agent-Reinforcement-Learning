python multi_agent_train.py# multi_agent_evaluate.py — Evaluation for GridEnv4 Multi-Agent System

import pickle
import random
from env_grid4 import GridEnv4
from controller import MasterAgent
from agent import QAgent
from utils import EPISODE_LENGTH


# ---------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------

def load_all(filename="multi_qtables.pkl", seed=0):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # Load local agents
    local_agents = []
    for item in data["locals"]:
        ag = QAgent(seed)
        ag.Q1.update(item["Q1"])
        ag.Q2.update(item["Q2"])
        local_agents.append(ag)

    # Load master agent
    master = MasterAgent(seed)
    master.Q.update(data["master_Q"])

    print("[Load] Multi-agent Q-tables loaded.")
    return local_agents, master


# ---------------------------------------------------------
# RUN ONE EVALUATION EPISODE
# ---------------------------------------------------------

def run_episode(local_agents, master, seed):
    env = GridEnv4(seed)
    states = env.reset()
    done = False

    total_wait = 0
    total_pass = 0

    while not done:

        global_state = tuple(states)

        # Master greedy action
        q0 = master.Q[(global_state, 0)]
        q1 = master.Q[(global_state, 1)]
        master_action = 0 if q0 >= q1 else 1

        # Local greedy actions
        local_actions = []
        for i in range(4):
            state_i = states[i]

            # best local action
            values = [
                0.5 * (local_agents[i].Q1[(state_i, a)] + local_agents[i].Q2[(state_i, a)])
                for a in range(4)
            ]
            best_a = values.index(max(values))
            local_actions.append(best_a)

        # Step environment
        outputs, done = env.step(local_actions)

        # Collect stats
        for out in outputs:
            total_pass += out["passed"]
            total_wait += out["wait_delta"]

        states = [o["state"] for o in outputs]

    return total_wait / EPISODE_LENGTH, total_pass


# ---------------------------------------------------------
# FIXED TIMER BASELINE
# ---------------------------------------------------------

def run_fixed_timer(seed):
    env = GridEnv4(seed)
    states = env.reset()
    done = False

    total_wait = 0
    total_pass = 0
    t = 0

    while not done:
        phase = (t // 20) % 4
        local_actions = [phase] * 4

        outputs, done = env.step(local_actions)

        for out in outputs:
            total_pass += out["passed"]
            total_wait += out["wait_delta"]

        states = [o["state"] for o in outputs]
        t += 1

    return total_wait / EPISODE_LENGTH, total_pass


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main():

    locals, master = load_all()

    seeds = [6000 + i for i in range(30)]

    rl_waits = []
    rl_passes = []
    base_waits = []
    base_passes = []

    print("\nEvaluating multi-agent system...\n")

    for s in seeds:
        w, p = run_episode(locals, master, s)
        rl_waits.append(w)
        rl_passes.append(p)

        bw, bp = run_fixed_timer(s)
        base_waits.append(bw)
        base_passes.append(bp)

    avg_rl_wait = sum(rl_waits) / len(rl_waits)
    avg_rl_pass = sum(rl_passes) / len(rl_passes)
    avg_base_wait = sum(base_waits) / len(base_waits)
    avg_base_pass = sum(base_passes) / len(base_passes)

    print("==============================================")
    print("         FINAL PERFORMANCE REPORT")
    print("==============================================")
    print(f"Baseline Avg Wait:      {avg_base_wait:.2f}")
    print(f"Baseline Vehicles Pass: {avg_base_pass:.1f}")
    print("----------------------------------------------")
    print(f"RL Avg Wait:            {avg_rl_wait:.2f}")
    print(f"RL Vehicles Pass:       {avg_rl_pass:.1f}")
    print("----------------------------------------------")
    improvement = (avg_base_wait - avg_rl_wait) / avg_base_wait * 100
    print(f"Improvement:            {improvement:.2f}%")
    print("==============================================\n")


if __name__ == "__main__":
    main()
