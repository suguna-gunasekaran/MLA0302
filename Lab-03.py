import random
import math

# -------------------------------
# Price Options (Arms)
# -------------------------------
prices = [100, 150, 200, 250]

# Probability that a customer buys at each price
purchase_prob = [0.8, 0.6, 0.4, 0.2]

customers = 500


# Simulate a customer purchase
def get_reward(arm):
    if random.random() < purchase_prob[arm]:
        return prices[arm]
    return 0


# ===========================================
# 1. Epsilon-Greedy
# ===========================================
epsilon = 0.1

counts = [0] * len(prices)
values = [0] * len(prices)
total_reward = 0

for i in range(customers):

    if random.random() < epsilon:
        arm = random.randint(0, len(prices)-1)
    else:
        arm = values.index(max(values))

    reward = get_reward(arm)

    counts[arm] += 1
    total_reward += reward

    values[arm] += (reward - values[arm]) / counts[arm]

epsilon_reward = total_reward


# ===========================================
# 2. UCB
# ===========================================
counts = [0] * len(prices)
values = [0] * len(prices)
total_reward = 0

for i in range(customers):

    if 0 in counts:
        arm = counts.index(0)
    else:
        ucb_values = []

        for j in range(len(prices)):
            bonus = math.sqrt((2 * math.log(i + 1)) / counts[j])
            ucb_values.append(values[j] + bonus)

        arm = ucb_values.index(max(ucb_values))

    reward = get_reward(arm)

    counts[arm] += 1
    total_reward += reward

    values[arm] += (reward - values[arm]) / counts[arm]

ucb_reward = total_reward


# ===========================================
# 3. Thompson Sampling
# ===========================================
success = [1] * len(prices)
failure = [1] * len(prices)

total_reward = 0

for i in range(customers):

    samples = []

    for j in range(len(prices)):
        samples.append(random.betavariate(success[j], failure[j]))

    arm = samples.index(max(samples))

    reward = get_reward(arm)

    total_reward += reward

    if reward > 0:
        success[arm] += 1
    else:
        failure[arm] += 1

thompson_reward = total_reward


# ===========================================
# Display Results
# ===========================================
print("Dynamic Pricing using Multi-Armed Bandits")
print("----------------------------------------")
print("Customers:", customers)

print("\nTotal Revenue")
print("--------------")
print("Epsilon-Greedy :", epsilon_reward)
print("UCB            :", ucb_reward)
print("Thompson Samp. :", thompson_reward)

# Best Strategy
revenues = {
    "Epsilon-Greedy": epsilon_reward,
    "UCB": ucb_reward,
    "Thompson Sampling": thompson_reward
}

best = max(revenues, key=revenues.get)

print("\nBest Strategy:", best)
