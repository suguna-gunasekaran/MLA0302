import numpy as np
import random

# Sample stock prices
prices = [100, 105, 102, 108, 110, 107, 115]

# Q-tables (Online and Target)
online_q = np.zeros((len(prices), 3))   # Hold, Buy, Sell
target_q = np.zeros((len(prices), 3))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(100):

    holding = False

    for state in range(len(prices) - 1):

        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(online_q[state])

        reward = 0

        # Buy
        if action == 1 and not holding:
            holding = True

        # Sell
        elif action == 2 and holding:
            reward = prices[state + 1] - prices[state]
            holding = False

        # Double DQN Update
        best_action = np.argmax(online_q[state + 1])
        target = reward + gamma * target_q[state + 1][best_action]

        online_q[state][action] += alpha * (
            target - online_q[state][action]
        )

    # Update Target Network
    target_q = np.copy(online_q)

print("Learned Q-Table:")
print(online_q)

actions = ["Hold", "Buy", "Sell"]
print("\nBest action at each day:")
for i in range(len(prices)):
    print("Day", i + 1, "Price =", prices[i], "->", actions[np.argmax(online_q[i])])
