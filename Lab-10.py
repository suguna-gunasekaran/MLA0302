import random

# User Inputs
policy = float(input("Enter Initial Investment Probability (0 to 1): "))
learning_rate = float(input("Enter Learning Rate (e.g., 0.05): "))
episodes = int(input("Enter Number of Episodes: "))

total_reward = 0

print("\nInvestment Strategy using Policy Gradient")
print("-----------------------------------------")

for episode in range(1, episodes + 1):

    # Select Action
    if random.random() < policy:
        action = "Invest"
    else:
        action = "Do Not Invest"

    # Simulate Market Return
    if action == "Invest":
        reward = random.choice([10, -5])   # Profit or Loss
    else:
        reward = 0

    total_reward += reward

    # Policy Gradient Update
    if reward > 0:
        policy = policy + learning_rate * (1 - policy)
    elif reward < 0:
        policy = policy - learning_rate * policy

    # Keep Probability Between 0 and 1
    policy = max(0, min(1, policy))

print("\nSimulation Completed")
print("--------------------")
print("Total Return :", total_reward)
print("Final Investment Probability :", round(policy, 2))
