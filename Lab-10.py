import random

# Initial probability of investing
policy = 0.5

learning_rate = 0.05
episodes = 100

total_reward = 0

print("Investment Strategy using Policy Gradient")
print("-----------------------------------------")

for episode in range(1, episodes + 1):

    # Select action
    if random.random() < policy:
        action = "Invest"
    else:
        action = "Do Not Invest"

    # Simulate market return
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

    # Keep probability between 0 and 1
    policy = max(0, min(1, policy))

print("\nSimulation Completed")
print("--------------------")
print("Total Return :", total_reward)
print("Final Investment Probability :", round(policy, 2))
