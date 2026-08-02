import numpy as np
import random

# Number of Days
n = int(input("Enter Number of Stock Prices: "))

# Stock Prices
prices = []
for i in range(n):
    price = float(input(f"Enter Price for Day {i+1}: "))
    prices.append(price)

# Learning Parameters
alpha = float(input("Enter Learning Rate (e.g., 0.1): "))
gamma = float(input("Enter Discount Factor (e.g., 0.9): "))
epsilon = float(input("Enter Epsilon (e.g., 0.2): "))
episodes = int(input("Enter Number of Episodes: "))

# Q-Tables (Online and Target)
online_q = np.zeros((len(prices), 3))   # Hold, Buy, Sell
target_q = np.zeros((len(prices), 3))

# Training
for episode in range(episodes):

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

# Results
print("\nLearned Q-Table:")
print(online_q)

actions = ["Hold", "Buy", "Sell"]

print("\nBest Action for Each Day:")
for i in range(len(prices)):
    print(f"Day {i+1} | Price = {prices[i]} -> {actions[np.argmax(online_q[i])]}")
