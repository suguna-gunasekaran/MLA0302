# A robot vacuum cleaner navigates a house with various rooms and obstacles. Use the SARSA
# algorithm to learn the optimal cleaning policy that maximizes the cleaned area while
# minimizing energy usage. Implement this in Python.

import numpy as np

# User Inputs
rooms = int(input("Enter Number of Rooms: "))
actions_count = 2  # Clean, Move

alpha = float(input("Enter Learning Rate (e.g., 0.1): "))
gamma = float(input("Enter Discount Factor (e.g., 0.9): "))
episodes = int(input("Enter Number of Episodes: "))

# Reward Room
reward_room = int(input(f"Enter Reward Room (1 to {rooms}): ")) - 1

# Reward Values
positive_reward = float(input("Enter Reward for Target Room: "))
negative_reward = float(input("Enter Reward for Other Rooms: "))

# Initialize Q-Table
q = np.zeros((rooms, actions_count))

# SARSA Training
for episode in range(episodes):

    for state in range(rooms - 1):

        action = np.argmax(q[state])

        if state == reward_room:
            reward = positive_reward
        else:
            reward = negative_reward

        next_state = state + 1
        next_action = np.argmax(q[next_state])

        # SARSA Update
        q[state][action] = q[state][action] + alpha * (
            reward + gamma * q[next_state][next_action] - q[state][action]
        )

# Display Q-Table
print("\nQ-Table")
print(q)

actions = ["Clean", "Move"]

print("\nBest Action")
for i in range(rooms):
    print(f"Room {i+1}: {actions[np.argmax(q[i])]}")
