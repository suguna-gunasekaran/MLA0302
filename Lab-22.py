# ============================================================
# K-ARMED BANDIT FOR MARKETING CAMPAIGN OPTIMIZATION
# EPSILON-GREEDY, UCB AND THOMPSON SAMPLING
# ============================================================

import numpy as np
import random
import math

# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))
steps = int(input("Enter number of steps per episode: "))

num_arms = int(input("Enter number of marketing campaigns: "))

epsilon = float(input("Enter initial epsilon: "))
epsilon_decay = float(input("Enter epsilon decay rate: "))

ucb_c = float(input("Enter UCB exploration constant: "))

alpha_start = float(input(
    "Enter initial Alpha for Thompson Sampling: "
))

beta_start = float(input(
    "Enter initial Beta for Thompson Sampling: "
))


# ---------------- CAMPAIGNS ----------------

campaigns = [
    "Email Campaign",
    "Social Media Campaign",
    "SMS Campaign",
    "Online Advertisement",
    "Search Advertisement",
    "Influencer Campaign"
]

# Use only required number of campaigns
campaigns = campaigns[:num_arms]

# True customer response probabilities
true_probability = [
    0.60,
    0.80,
    0.50,
    0.70,
    0.65,
    0.75
]

true_probability = true_probability[:num_arms]


# ---------------- ENVIRONMENT ----------------

def environment_step(action):

    if random.random() < true_probability[action]:
        return 1
    else:
        return 0


total_steps = episodes * steps


# ============================================================
# EPSILON-GREEDY
# ============================================================

print("\n==========================================")
print("EPSILON-GREEDY ALGORITHM")
print("==========================================")

epsilon_Q = np.zeros(num_arms)
epsilon_count = np.zeros(num_arms)

epsilon_reward = 0
current_epsilon = epsilon

for step in range(total_steps):

    # Exploration
    if random.random() < current_epsilon:

        action = random.randint(
            0,
            num_arms - 1
        )

    # Exploitation
    else:

        action = np.argmax(epsilon_Q)

    reward = environment_step(action)

    epsilon_count[action] += 1

    epsilon_Q[action] += (
        reward - epsilon_Q[action]
    ) / epsilon_count[action]

    epsilon_reward += reward

    current_epsilon = max(
        0.05,
        current_epsilon * epsilon_decay
    )


# ============================================================
# UCB ALGORITHM
# ============================================================

print("\n==========================================")
print("UCB ALGORITHM")
print("==========================================")

ucb_Q = np.zeros(num_arms)
ucb_count = np.zeros(num_arms)

ucb_reward = 0

for step in range(1, total_steps + 1):

    # Select each campaign once
    if step <= num_arms:

        action = step - 1

    else:

        ucb_values = []

        for i in range(num_arms):

            confidence = ucb_c * math.sqrt(
                math.log(step) /
                ucb_count[i]
            )

            ucb = ucb_Q[i] + confidence

            ucb_values.append(ucb)

        action = np.argmax(ucb_values)

    reward = environment_step(action)

    ucb_count[action] += 1

    ucb_Q[action] += (
        reward - ucb_Q[action]
    ) / ucb_count[action]

    ucb_reward += reward


# ============================================================
# THOMPSON SAMPLING
# ============================================================

print("\n==========================================")
print("THOMPSON SAMPLING")
print("==========================================")

alpha = np.full(
    num_arms,
    alpha_start
)

beta = np.full(
    num_arms,
    beta_start
)

thompson_reward = 0

for step in range(total_steps):

    samples = []

    for i in range(num_arms):

        sample = np.random.beta(
            alpha[i],
            beta[i]
        )

        samples.append(sample)

    action = np.argmax(samples)

    reward = environment_step(action)

    if reward == 1:

        alpha[action] += 1

    else:

        beta[action] += 1

    thompson_reward += reward


# ============================================================
# LEARNED VALUES
# ============================================================

print("\n==========================================")
print("LEARNED CAMPAIGN VALUES")
print("==========================================")

for i in range(num_arms):

    thompson_value = (
        alpha[i] /
        (alpha[i] + beta[i])
    )

    print(
        campaigns[i],
        "\n  True Probability:",
        true_probability[i],
        "\n  Epsilon-Greedy:",
        round(epsilon_Q[i], 3),
        "\n  UCB:",
        round(ucb_Q[i], 3),
        "\n  Thompson Sampling:",
        round(thompson_value, 3)
    )


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

print("\n==========================================")
print("PERFORMANCE COMPARISON")
print("==========================================")

print(
    "Epsilon-Greedy Total Reward:",
    epsilon_reward
)

print(
    "UCB Total Reward:",
    ucb_reward
)

print(
    "Thompson Sampling Total Reward:",
    thompson_reward
)


# ============================================================
# BEST ALGORITHM
# ============================================================

performance = {
    "Epsilon-Greedy": epsilon_reward,
    "UCB": ucb_reward,
    "Thompson Sampling": thompson_reward
}

best_algorithm = max(
    performance,
    key=performance.get
)


# ============================================================
# OPTIMAL CAMPAIGN
# ============================================================

best_campaign = np.argmax(
    true_probability
)


# ============================================================
# FINAL ANALYSIS
# ============================================================

print("\n==========================================")
print("FINAL ANALYSIS")
print("==========================================")

print(
    "Best Performing Algorithm:",
    best_algorithm
)

print(
    "Highest Total Reward:",
    performance[best_algorithm]
)

print(
    "Optimal Marketing Campaign:",
    campaigns[best_campaign]
)

print(
    "Expected Customer Response:",
    true_probability[best_campaign]
)

print(
    "Final Epsilon:",
    round(current_epsilon, 3)
)

print("\nK-Armed Bandit Marketing Optimization Completed.")
