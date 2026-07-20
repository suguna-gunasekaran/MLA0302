import random

# Customer Service Representatives
agents = ["Agent A", "Agent B", "Agent C"]

# Probability of successful call handling
success_prob = {
    "Agent A": 0.70,
    "Agent B": 0.85,
    "Agent C": 0.60
}

calls = 1000

# ---------------- Random Policy ----------------
random_reward = 0

for i in range(calls):

    agent = random.choice(agents)

    if random.random() < success_prob[agent]:
        random_reward += 1

random_value = random_reward / calls


# ---------------- Best-Agent Policy ----------------
best_reward = 0

best_agent = max(success_prob, key=success_prob.get)

for i in range(calls):

    if random.random() < success_prob[best_agent]:
        best_reward += 1

best_value = best_reward / calls


# ---------------- Results ----------------
print("Call Center Assignment using Monte Carlo Simulation")
print("---------------------------------------------------")
print("Number of Calls:", calls)

print("\nEstimated Value Function")
print("------------------------")
print("Random Policy     :", round(random_value, 3))
print("Best-Agent Policy :", round(best_value, 3))

if best_value > random_value:
    print("\nBest Policy: Best-Agent Policy")
else:
    print("\nBest Policy: Random Policy")
