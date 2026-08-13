# Model a smart grid that manages energy consumption and production to minimize costs and
# balance supply and demand using Trust Region Policy Optimization (TRPO) to optimize
# energy management.

import numpy as np

# Number of Time Slots
n = int(input("Enter Number of Time Slots: "))

# Energy Demand
demand = []
print("\nEnter Energy Demand:")
for i in range(n):
    demand.append(float(input(f"Demand at Time {i+1}: ")))

# Energy Production
production = []
print("\nEnter Energy Production:")
for i in range(n):
    production.append(float(input(f"Production at Time {i+1}: ")))

# Learning Parameters
learning_rate = float(input("\nEnter Learning Rate (e.g., 0.1): "))
episodes = int(input("Enter Number of Episodes: "))

# Actions
actions = ["Increase", "Maintain", "Decrease"]

# Initial Policy Probabilities
policy = np.array([[0.33, 0.34, 0.33] for _ in range(n)])

# TRPO Training (Simplified)
for episode in range(episodes):

    for state in range(n):

        # Select Action
        action = np.random.choice(3, p=policy[state])

        # Compute Balance
        if action == 0:
            balance = production[state] + 5
        elif action == 1:
            balance = production[state]
        else:
            balance = production[state] - 5

        # Reward
        reward = -abs(balance - demand[state])

        # Policy Update
        policy[state][action] += learning_rate * (reward / 100)

        # Normalize Probabilities
        policy[state] = np.clip(policy[state], 0.01, 0.98)
        policy[state] /= np.sum(policy[state])

# Display Final Policy
print("\nFinal Policy")

for i in range(n):
    print(f"Time {i+1}: {policy[i]}")

# Display Best Action
print("\nBest Energy Action")

for i in range(n):
    print(f"Time {i+1} -> {actions[np.argmax(policy[i])]}")
