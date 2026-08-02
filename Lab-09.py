import random

# Number of Customer Service Representatives
n = int(input("Enter Number of Agents: "))

agents = []
success_prob = {}

# Input Agent Names and Success Probabilities
for i in range(n):
    name = input(f"Enter Name of Agent {i+1}: ")
    probability = float(input(f"Enter Success Probability for {name} (0 to 1): "))
    agents.append(name)
    success_prob[name] = probability

# Number of Calls
calls = int(input("Enter Number of Calls: "))

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
print("\nCall Center Assignment using Monte Carlo Simulation")
print("---------------------------------------------------")
print("Number of Calls:", calls)
print("Best Agent:", best_agent)

print("\nEstimated Value Function")
print("------------------------")
print("Random Policy     :", round(random_value, 3))
print("Best-Agent Policy :", round(best_value, 3))

if best_value > random_value:
    print("\nBest Policy: Best-Agent Policy")
else:
    print("\nBest Policy: Random Policy")
