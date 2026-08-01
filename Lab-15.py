import numpy as np
import random

# States: Waiting calls
states = 5

# Q-table (2 actions: Agent1, Agent2)
q = np.zeros((states, 2))

alpha = 0.1

for episode in range(100):

    for state in range(states):

        action = random.randint(0, 1)

        # Reward
        if action == 0:
            reward = 10      # Faster service
        else:
            reward = 5       # Slower service

        # Monte Carlo Update
        q[state][action] = q[state][action] + alpha * (
            reward - q[state][action]
        )

print("Q-Table")
print(q)

agents = ["Agent 1", "Agent 2"]

print("\nBest Agent")
for i in range(states):
    print("State", i + 1, ":", agents[np.argmax(q[i])])
