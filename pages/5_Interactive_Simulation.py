import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, os

# Add root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from env import TrafficEnvMixed
from agent import QAgent

st.title("🧪 Interactive Simulation – Adjust Traffic Density")
st.markdown("---")

# -----------------------------
# Load trained agent
# -----------------------------
try:
    with open(os.path.join(ROOT, "qtable.pkl"), "rb") as f:
        data = pickle.load(f)
    agent = QAgent()
    agent.Q1.update(data["Q1"])
    agent.Q2.update(data["Q2"])
    st.success("RL Agent Loaded Successfully!")
except:
    st.error("qtable.pkl not found! Train the RL model first.")
    st.stop()

# -----------------------------
# User input: Traffic Density
# -----------------------------
st.subheader("🚦 Select Traffic Density")

density_choice = st.radio(
    "Choose traffic level:",
    ["Low", "Medium", "High", "Custom"]
)

if density_choice == "Low":
    spawn_prob = 0.10
elif density_choice == "Medium":
    spawn_prob = 0.20
elif density_choice == "High":
    spawn_prob = 0.35
else:
    spawn_prob = st.slider("Custom spawn probability:", 0.05, 0.60, 0.20)

st.info(f"Vehicle spawn rate selected: **{spawn_prob}**")

# Override global SPAWN_PROB
import utils
utils.SPAWN_PROB = spawn_prob

# -----------------------------
# Run Simulation Button
# -----------------------------
run = st.button("▶ Run Simulation")

if run:

    # ----- BASELINE EVALUATION -----
    base_env = TrafficEnvMixed(seed=123)
    base_state = base_env.reset()

    base_wait, base_pass = 0, 0
    done = False
    while not done:
        action = (base_env.t // 20) % 4  # fixed timer
        base_state, r, done, info = base_env.step(action)
        base_wait += info["total_waiting"]
        base_pass += info["passed"]

    base_avg_wait = base_wait / base_env.t

    # ----- RL MODEL EVALUATION -----
    rl_env = TrafficEnvMixed(seed=123)
    rl_state = rl_env.reset()

    rl_wait, rl_pass = 0, 0
    done = False
    while not done:
        action = agent.best_action(rl_state)
        rl_state, r, done, info = rl_env.step(action)
        rl_wait += info["total_waiting"]
        rl_pass += info["passed"]

    rl_avg_wait = rl_wait / rl_env.t

    # ----- OUTPUT -----
    st.subheader("📊 Simulation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Waiting Time", f"{base_avg_wait:.2f}s")
    col2.metric("RL Waiting Time", f"{rl_avg_wait:.2f}s")
    improvement = (base_avg_wait - rl_avg_wait) / base_avg_wait * 100
    col3.metric("Improvement", f"{improvement:.2f}%")

    st.write("### 🚗 Vehicles Passed")
    st.write(f"Baseline: **{base_pass}**")
    st.write(f"RL Model: **{rl_pass}**")

    # ----- Chart -----
    fig, ax = plt.subplots()
    ax.bar(["Baseline", "RL"], [base_avg_wait, rl_avg_wait], color=["red", "green"])
    ax.set_ylabel("Avg Waiting Time (seconds)")
    st.pyplot(fig)
