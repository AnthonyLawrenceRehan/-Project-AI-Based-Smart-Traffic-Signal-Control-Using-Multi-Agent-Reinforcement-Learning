# train.py — Final Optimized Training Script for Double Q-Learning Traffic Agent

import pickle
import random
from agent import QAgent
from env import TrafficEnvMixed
from utils import EPISODES, SEED


# ------------------------------------------------------------
# SAVE / LOAD
# ------------------------------------------------------------
def save_agent(agent, filename="qtable.pkl"):
    data = {"Q1": dict(agent.Q1), "Q2": dict(agent.Q2)}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("[Save] Q-tables saved.")


def load_agent(filename="qtable.pkl", seed=0):
    agent = QAgent(seed=seed)
    with open(filename, "rb") as f:
        data = pickle.load(f)
    agent.Q1.update(data["Q1"])
    agent.Q2.update(data["Q2"])
    print("[Load] Q-tables loaded.")
    return agent


# ------------------------------------------------------------
# TRAINING LOOP (FINAL & CORRECTED)
# ------------------------------------------------------------
def train_agent(episodes=1000, seed=0):
    env = TrafficEnvMixed(seed=seed)
    agent = QAgent(seed=seed)

    # Start fully exploratory
    agent.eps = 1.0

    # Epsilon decay parameters
    agent.eps_decay = 0.997
    agent.eps_min = 0.01

    print("Initial EPS:", agent.eps)

    for ep in range(1, episodes + 1):

        # 🔥 FIX: update epsilon ONCE per episode
        agent.update_epsilon()

        state = env.reset()
        done = False

        while not done:

            # Choose action with updated epsilon
            action = agent.choose_action(state, ep)

            # SAFETY — avoid switching too early
            if action != env.phase and env.time_in_phase < 3:
                action = env.phase

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Clip reward
            reward = max(min(reward, 10), -10)

            # Update Q-tables
            agent.update(state, action, reward, next_state, ep)

            state = next_state

        # Log progress
        if ep % 50 == 0:
            print(f"[Train] Ep {ep}/{episodes}  eps={agent.eps:.3f}")

    return agent


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
def evaluate_agent(agent, seeds):
    waits, passes = [], []

    for s in seeds:
        env = TrafficEnvMixed(seed=s)
        agent.random = random.Random(s)

        state = env.reset()
        done = False
        total_wait, total_pass = 0, 0

        while not done:
            action = agent.best_action(state)
            state, reward, done, info = env.step(action)
            total_wait += info["total_waiting"]
            total_pass += info["passed"]

        waits.append(total_wait / env.t)
        passes.append(total_pass)

    return sum(waits) / len(waits), sum(passes) / len(passes)


def evaluate_fixed_timer(seeds):
    waits, passes = [], []

    for s in seeds:
        env = TrafficEnvMixed(seed=s)
        state = env.reset()
        done = False
        t = 0
        total_wait, total_pass = 0, 0

        while not done:
            action = (t // 20) % 4   # fixed-time cycle
            state, reward, done, info = env.step(action)
            total_wait += info["total_waiting"]
            total_pass += info["passed"]
            t += 1

        waits.append(total_wait / env.t)
        passes.append(total_pass)

    return sum(waits) / len(waits), sum(passes) / len(passes)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("Training started...")
    agent = train_agent(EPISODES, SEED)
    save_agent(agent)

    seeds = [1000 + i for i in range(30)]
    print("Evaluating on same seeds...")

    b_wait, b_pass = evaluate_fixed_timer(seeds)
    a_wait, a_pass = evaluate_agent(agent, seeds)

    print(f"\nBaseline: wait={b_wait:.2f}, passed={b_pass:.1f}")
    print(f"Agent:    wait={a_wait:.2f}, passed={a_pass:.1f}")

    print(f"\nImprovement: {(b_wait - a_wait) / b_wait * 100:.2f}%")


if __name__ == "__main__":
    main()
