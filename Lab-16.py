import numpy as np
import random

# Energy demand and production
demand = [50, 60, 55, 70, 65]
production = [55, 58, 60, 68, 70]

# Actions
actions = ["Increase", "Maintain", "Decrease"]

# Policy probabilities
policy = np.array([[0.33, 0.34, 0.33] for _ in range(len(demand))])

learning_rate = 0.1

for episode in range(100):

    for state in range(len(demand)):

        action = np.random.choice(3, p=policy[state])

        # Reward based on balancing supply and demand
        if action == 0:
            balance = production[state] + 5
        elif action == 1:
            balance = production[state]
        else:
            balance = production[state] - 5

        reward = -abs(balance - demand[state])

        # Simplified TRPO Policy Update
        policy[state][action] += learning_rate * (reward / 100)

        policy[state] = np.clip(policy[state], 0.01, 0.98)
        policy[state] /= np.sum(policy[state])

print("Final Policy")

for i in range(len(demand)):
    print("Time", i + 1, ":", policy[i])

print("\nBest Energy Action")

for i in range(len(demand)):
    print("Time", i + 1, "->", actions[np.argmax(policy[i])])
