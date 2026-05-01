# 🚦 Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control

 📌 Abstract
Traffic congestion remains a major challenge in urban environments due to inefficient fixed-time signal control systems. This project proposes a multi-agent reinforcement learning (RL) framework for adaptive traffic signal control. Each intersection is modeled as an independent agent that learns optimal signal policies using Q-learning. A master coordination mechanism is introduced to improve traffic flow across multiple intersections. The system is evaluated in a custom simulation environment and demonstrates approximately 30% reduction in average vehicle waiting time compared to traditional fixed-time control.



## 🧠 1. Introduction
Conventional traffic signal systems rely on static timing plans that fail to adapt to dynamic traffic conditions. This leads to unnecessary delays, congestion, and inefficient utilization of road infrastructure.

This project aims to address these limitations using reinforcement learning, enabling traffic signals to:
- Learn from real-time traffic conditions
- Adapt signal timings dynamically
- Optimize traffic flow across multiple intersections

---

## ⚙️ 2. System Architecture

The system consists of the following components:

### 🔹 2.1 Environment (Simulation Layer)
- Custom Gym-like traffic simulation
- Models vehicle queues in four directions (N, S, E, W)
- Supports stochastic vehicle generation
- Implements signal phases and traffic flow rules
- Multi-intersection grid (4-node topology)

### 🔹 2.2 RL Agent (Decision Layer)
- Q-learning based agent
- Learns optimal signal phase selection
- Uses epsilon-greedy exploration strategy
- Stores policy in Q-table

### 🔹 2.3 Multi-Agent System
- Each intersection acts as an independent RL agent
- Agents operate simultaneously in a shared environment

### 🔹 2.4 Master Controller
- Observes global traffic conditions
- Improves coordination between intersections
- Reduces downstream congestion

### 🔹 2.5 Visualization Layer
- Built using Streamlit
- Provides interactive simulation and performance comparison
- Displays metrics and traffic flow behavior

---

## 🧮 3. Methodology

### 🔹 3.1 State Representation
Each state is defined as:
- Discretized queue lengths (per direction)
- Front vehicle type (straight / left turn)
- Current signal phase

### 🔹 3.2 Action Space
The agent selects one of the signal phases:
- NS straight
- EW straight
- NS left-turn
- EW left-turn

### 🔹 3.3 Reward Function
The reward is based on traffic efficiency:
- Penalizes increase in waiting time
- Encourages vehicle throughput
- Minimizes congestion

### 🔹 3.4 Learning Algorithm
Q-learning update rule:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

---

## 📊 4. Experimental Setup

- Number of intersections: 4
- Training episodes: 4000
- Traffic generation: stochastic (based on spawn probability)
- Simulation length: configurable episode duration

---

## 📈 5. Results

| Method        | Avg Waiting Time | Queue Length | Throughput |
|--------------|----------------|-------------|-----------|
| Fixed-Time   | ~55s           | High        | Low       |
| RL (Proposed)| ~38s           | Reduced     | Improved  |

### 🔥 Key Observations:
- ~30% reduction in waiting time
- Better traffic flow distribution
- Improved adaptability under varying traffic conditions

---
