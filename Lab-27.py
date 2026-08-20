import numpy as np
import random

# Number of Stocks
n = int(input("Enter Number of Stocks: "))

stocks = []
returns = []

for i in range(n):
    name = input(f"Enter Stock {i+1} Name: ")
    ret = float(input(f"Enter Expected Return for {name}: "))
    stocks.append(name)
    returns.append(ret)

learning_rate = float(input("Enter Learning Rate: "))
episodes = int(input("Enter Number of Episodes: "))

# Actor and Critic
actor = np.random.rand(n)
critic = np.zeros(n)

for episode in range(episodes):

    for state in range(n):

        reward = returns[state]

        # Critic Update
        critic[state] += learning_rate * (reward - critic[state])

        # Actor Update
        actor[state] += learning_rate * critic[state] / 100

print("\nInvestment Scores")

for i in range(n):
    print(stocks[i], ":", round(actor[i], 2))

print("\nRecommended Portfolio")

order = np.argsort(actor)[::-1]

for i in order:
    print(stocks[i], "Score =", round(actor[i], 2))
