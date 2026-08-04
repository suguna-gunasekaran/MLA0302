# Implement Q-learning to develop an AI agent that plays a simple grid-based game (e.g., a
# basic version of Pac-Man). The agent should learn to collect rewards (e.g., food) and avoid
# penalties (e.g., ghosts). Write a Python program to train and evaluate the AI agent

import numpy as np

# User Inputs
states = int(input("Enter Number of States: "))
alpha = float(input("Enter Learning Rate (e.g., 0.1): "))
gamma = float(input("Enter Discount Factor (e.g., 0.9): "))
episodes = int(input("Enter Number of Episodes: "))

goal_state = int(input(f"Enter Goal State (0 to {states-1}): "))
goal_reward = float(input("Enter Goal Reward: "))
step_reward = float(input("Enter Step Reward: "))

# Initialize Q-Table
q = np.zeros((states, 2))   # Actions: Left, Right

# Q-Learning Training
for episode in range(episodes):

    state = 0

    while state < states - 1:

        # Choose Best Action
        action = np.argmax(q[state])

        # Move to Next State
        next_state = state + 1

        # Reward
        if next_state == goal_state:
            reward = goal_reward
        else:
            reward = step_reward

        # Q-Learning Update
        q[state][action] = q[state][action] + alpha * (
            reward + gamma * np.max(q[next_state]) - q[state][action]
        )

        state = next_state

# Display Q-Table
print("\nQ-Table")
print(q)

actions = ["Left", "Right"]

print("\nBest Action")
for i in range(states):
    print(f"State {i}: {actions[np.argmax(q[i])]}")
