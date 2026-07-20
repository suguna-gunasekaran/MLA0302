import random
import math

# ---------------------------
# Advertisements (Arms)
# ---------------------------
ads = ["Ad A", "Ad B", "Ad C", "Ad D"]

# Probability of click for each ad
click_prob = [0.15, 0.25, 0.35, 0.20]

users = 1000


# Simulate user click
def get_reward(ad):
    if random.random() < click_prob[ad]:
        return 1
    return 0


# =====================================
# 1. Epsilon-Greedy
# =====================================
epsilon = 0.1

counts = [0] * len(ads)
values = [0] * len(ads)
clicks = 0

for i in range(users):

    if random.random() < epsilon:
        arm = random.randint(0, len(ads)-1)
    else:
        arm = values.index(max(values))

    reward = get_reward(arm)

    counts[arm] += 1
    clicks += reward

    values[arm] += (reward - values[arm]) / counts[arm]

epsilon_ctr = clicks / users


# =====================================
# 2. UCB
# =====================================
counts = [0] * len(ads)
values = [0] * len(ads)
clicks = 0

for i in range(users):

    if 0 in counts:
        arm = counts.index(0)
    else:
        ucb = []

        for j in range(len(ads)):
            bonus = math.sqrt((2 * math.log(i + 1)) / counts[j])
            ucb.append(values[j] + bonus)

        arm = ucb.index(max(ucb))

    reward = get_reward(arm)

    counts[arm] += 1
    clicks += reward

    values[arm] += (reward - values[arm]) / counts[arm]

ucb_ctr = clicks / users


# =====================================
# 3. Thompson Sampling
# =====================================
success = [1] * len(ads)
failure = [1] * len(ads)

clicks = 0

for i in range(users):

    samples = []

    for j in range(len(ads)):
        samples.append(random.betavariate(success[j], failure[j]))

    arm = samples.index(max(samples))

    reward = get_reward(arm)

    clicks += reward

    if reward == 1:
        success[arm] += 1
    else:
        failure[arm] += 1

thompson_ctr = clicks / users


# =====================================
# Results
# =====================================
print("Advertisement Selection using Multi-Armed Bandits")
print("-------------------------------------------------")
print("Users:", users)

print("\nClick-Through Rate (CTR)")
print("-------------------------")
print("Epsilon-Greedy :", round(epsilon_ctr, 3))
print("UCB            :", round(ucb_ctr, 3))
print("Thompson Samp. :", round(thompson_ctr, 3))

ctr = {
    "Epsilon-Greedy": epsilon_ctr,
    "UCB": ucb_ctr,
    "Thompson Sampling": thompson_ctr
}

best = max(ctr, key=ctr.get)

print("\nBest Algorithm:", best)
