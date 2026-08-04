# A call center uses Monte Carlo methods to optimize the assignment of customer service
# representatives to incoming calls. Implement Monte Carlo policy control in Python to
# minimize average call handling time.

import numpy as np
import random

# User Inputs
states = int(input("Enter Number of States: "))
alpha = float(input("Enter Learning Rate (e.g., 0.1): "))
episodes = int(input("Enter Number of Episodes: "))

reward_agent1 = float(input("Enter Reward for Agent 1: "))
reward_agent2 = float(input("Enter Reward for Agent 2: "))

# Q-table (2 actions: Agent 1, Agent 2)
q = np.zeros((states, 2))

# Monte Carlo Training
for episode in range(episodes):

    for state in range(states):

        # Randomly choose an agent
        action = random.randint(0, 1)

        # Reward
        if action == 0:
            reward = reward_agent1
        else:
            reward = reward_agent2

        # Monte Carlo Update
        q[state][action] = q[state][action] + alpha * (
            reward - q[state][action]
        )

# Display Q-Table
print("\nQ-Table")
print(q)

agents = ["Agent 1", "Agent 2"]

print("\nBest Agent for Each State")
for i in range(states):
    print(f"State {i+1}: {agents[np.argmax(q[i])]}")
