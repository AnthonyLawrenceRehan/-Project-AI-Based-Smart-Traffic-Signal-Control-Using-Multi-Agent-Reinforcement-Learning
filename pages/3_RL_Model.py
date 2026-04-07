import streamlit as st
import matplotlib.pyplot as plt

st.title("🧠 RL Model — Learning Behavior")
st.markdown("---")

st.write("### 📉 Epsilon Decay (Exploration → Exploitation)")

eps_values = []
eps = 1.0
decay = 0.997
min_eps = 0.02

for i in range(4000):
    eps = max(min_eps, eps * decay)
    eps_values.append(eps)

fig, ax = plt.subplots()
ax.plot(eps_values)
ax.set_title("Epsilon Decay")
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon Value")
st.pyplot(fig)

st.markdown("### Reward Function Used:")
st.code("""
Reward = 5 * passed_vehicles 
         - 0.2 * wait_time 
         - 2 * blocked_lanes 
         - 1 * switch_penalty
""")
