# multi_agent_train.py — Hierarchical Multi-Agent RL
# 4 Local Q-learning Agents + 1 Master Controller

import random
import pickle
from agent import QAgent
from controller import MasterAgent
from env_grid4 import GridEnv4
from utils import EPISODES, SEED


# ---------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------

def save_all(local_agents, master, filename="multi_qtables.pkl"):
    data = {
        "locals": [
            {"Q1": dict(a.Q1), "Q2": dict(a.Q2)}
            for a in local_agents
        ],
        "master_Q": dict(master.Q)
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("[Save] Multi-agent Q-tables saved.")


def load_all(filename="multi_qtables.pkl", seed=0):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    local_agents = []
    for lq in data["locals"]:
        ag = QAgent(seed=seed)
        ag.Q1.update(lq["Q1"])
        ag.Q2.update(lq["Q2"])
        local_agents.append(ag)

    master = MasterAgent(seed)
    master.Q.update(data["master_Q"])

    print("[Load] Multi-agent Q-tables loaded.")
    return local_agents, master


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------

def train_multiagent(episodes=1000, seed=0):
    env = GridEnv4(seed)

    # 4 independent RL agents
    locals = [QAgent(seed + i) for i in range(4)]

    # 1 master agent
    master = MasterAgent(seed + 99)

    for ep in range(1, episodes + 1):

        states = env.reset()
        done = False

        # ---------------------------------------------------
        # Update MASTER exploration ONCE per episode
        # ---------------------------------------------------
        master.update_epsilon()
        master.update_alpha()

        while not done:

            global_state = tuple(states)

            # Master chooses high-level mode (no epsilon update here)
            master_action = master.choose_action(global_state)

            # Local actions
            local_actions = []
            for i in range(4):

                # Local agent chooses action
                a = locals[i].choose_action(states[i], ep)

                # Master bias
                if master_action == 0:
                    pref = [0, 2]  # NS phases
                else:
                    pref = [1, 3]  # EW phases

                if random.random() < 0.25:
                    a = random.choice(pref)

                local_actions.append(a)

            # Step environment
            outputs, done = env.step(local_actions)

            # Local rewards
            local_rewards = []
            total_wait = 0
            total_pas = 0

            for i in range(4):
                passed = outputs[i]["passed"]
                wait_delta = outputs[i]["wait_delta"]

                reward = 5 * passed - 0.2 * wait_delta
                reward = max(min(reward, 10), -10)

                local_rewards.append(reward)
                total_wait += wait_delta
                total_pas += passed

            # Master reward
            global_reward = total_pas - 0.1 * total_wait

            next_states = [o["state"] for o in outputs]
            next_global_state = tuple(next_states)

            # Update master
            master.update(global_state, master_action, global_reward, next_global_state)

            # Update locals
            for i in range(4):
                locals[i].update(states[i], local_actions[i], local_rewards[i], next_states[i], ep)

            states = next_states

        if ep % 50 == 0:
            print(f"[Train {ep}/{episodes}] Master_eps={master.eps:.3f}")

    return locals, master


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Training 4-intersection multi-agent system...")
    local_agents, master = train_multiagent(EPISODES, SEED)
    save_all(local_agents, master)


if __name__ == "__main__":
    main()
