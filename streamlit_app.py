import streamlit as st

st.set_page_config(
    page_title="Smart Traffic RL Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Smart Traffic Control System")
st.markdown("### Reinforcement Learning Based Adaptive Signal Optimization")
st.markdown("---")

st.write("""
Welcome to the Smart Traffic Dashboard.

Use the **left sidebar** to navigate:
- **Home** → Overview of the project  
- **Baseline vs RL** → Comparison between normal fixed-timer and RL  
- **RL Model** → How Q-learning & epsilon decay works  
- **Network** → Multi-agent 4-intersection grid  
""")
