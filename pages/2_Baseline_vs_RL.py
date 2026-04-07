import streamlit as st
import matplotlib.pyplot as plt

st.title("⚖️ Baseline vs RL Performance")
st.markdown("---")

baseline_wait = 55.7
rl_wait = 38.5
improvement = (baseline_wait - rl_wait) / baseline_wait * 100

col1, col2, col3 = st.columns(3)

col1.metric("Baseline Waiting Time", f"{baseline_wait:.2f}s")
col2.metric("RL Waiting Time", f"{rl_wait:.2f}s")
col3.metric("Improvement", f"{improvement:.2f}%")

st.markdown("### 📉 Comparison Chart")

fig, ax = plt.subplots()
ax.bar(["Baseline", "Reinforcement Learning"], [baseline_wait, rl_wait], color=["red", "green"])
ax.set_ylabel("Average Waiting Time (seconds)")
st.pyplot(fig)
