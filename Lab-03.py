import random
import math

# -------------------------------
# User Input
# -------------------------------

n = int(input("Enter number of price options: "))

prices = []
purchase_prob = []

for i in range(n):
    price = int(input(f"Enter Price {i+1}: "))
    prob = float(input(f"Enter Purchase Probability for Price {price} (0 to 1): "))
    prices.append(price)
    purchase_prob.append(prob)

customers = int(input("Enter number of customers: "))
epsilon = float(input("Enter epsilon value (e.g., 0.1): "))


# Simulate Customer Purchase
def get_reward(arm):
    if random.random() < purchase_prob[arm]:
        return prices[arm]
    return 0


# ===========================================
# 1. Epsilon-Greedy
# ===========================================
counts = [0] * n
values = [0] * n
total_reward = 0

for i in range(customers):

    if random.random() < epsilon:
        arm = random.randint(0, n - 1)
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
counts = [0] * n
values = [0] * n
total_reward = 0

for i in range(customers):

    if 0 in counts:
        arm = counts.index(0)
    else:
        ucb_values = []

        for j in range(n):
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
success = [1] * n
failure = [1] * n

total_reward = 0

for i in range(customers):

    samples = []

    for j in range(n):
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
print("\nDynamic Pricing using Multi-Armed Bandits")
print("----------------------------------------")
print("Customers:", customers)

print("\nTotal Revenue")
print("--------------")
print("Epsilon-Greedy :", epsilon_reward)
print("UCB            :", ucb_reward)
print("Thompson Samp. :", thompson_reward)

revenues = {
    "Epsilon-Greedy": epsilon_reward,
    "UCB": ucb_reward,
    "Thompson Sampling": thompson_reward
}

best = max(revenues, key=revenues.get)

print("\nBest Strategy:", best)
