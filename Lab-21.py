# ============================================================
# RL FRAMEWORK FOR STREAMING CONTENT RECOMMENDATION
# USING UPPER CONFIDENCE BOUND (UCB)
# ============================================================

import numpy as np
import random
import math

# ---------------- INPUT ----------------

episodes = int(input("Enter number of episodes: "))
steps = int(input("Enter number of steps per episode: "))

print("\nStreaming Content:")
print("0 - Movies")
print("1 - Web Series")
print("2 - Sports")
print("3 - Documentaries")


# ---------------- ENVIRONMENT ----------------

contents = {
    0: "Movies",
    1: "Web Series",
    2: "Sports",
    3: "Documentaries"
}

num_actions = 4

# Probability that a user watches the selected content
watch_probability = {
    0: 0.70,
    1: 0.85,
    2: 0.60,
    3: 0.45
}


# ---------------- ENVIRONMENT STEP ----------------

def environment_step(action):

    if random.random() < watch_probability[action]:

        reward = 1

    else:

        reward = 0

    return reward


# ============================================================
# UCB ALGORITHM
# ============================================================

print("\n==========================================")
print("TRAINING UCB CONTENT RECOMMENDATION AGENT")
print("==========================================")

# Estimated reward of each content
Q = np.zeros(num_actions)

# Number of times each content was selected
count = np.zeros(num_actions)

total_reward = 0


# ============================================================
# UCB TRAINING
# ============================================================

for step in range(1, episodes * steps + 1):

    # --------------------------------------------------------
    # SELECT EACH CONTENT AT LEAST ONCE
    # --------------------------------------------------------

    if step <= num_actions:

        action = step - 1

    else:

        # ----------------------------------------------------
        # UCB FORMULA
        # UCB = Q + sqrt(2 * ln(t) / N)
        # ----------------------------------------------------

        ucb_values = []

        for i in range(num_actions):

            confidence = math.sqrt(
                2 * math.log(step) / count[i]
            )

            ucb = Q[i] + confidence

            ucb_values.append(ucb)

        action = np.argmax(ucb_values)

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    reward = environment_step(action)

    # Update count
    count[action] += 1

    # Update estimated reward
    Q[action] += (
        reward - Q[action]
    ) / count[action]

    total_reward += reward

    # Display progress
    if step == 1 or step % max(
        1,
        (episodes * steps) // 10
    ) == 0:

        print(
            "Step:",
            step,
            "| Reward:",
            reward,
            "| Total Reward:",
            total_reward
        )


# ============================================================
# UCB RESULTS
# ============================================================

print("\n==========================================")
print("UCB LEARNED VALUES")
print("==========================================")

for action in range(num_actions):

    print(
        contents[action],
        "-> Estimated Reward:",
        round(Q[action], 3),
        "| Selected:",
        int(count[action]),
        "times"
    )


# ============================================================
# BEST CONTENT
# ============================================================

best_action = np.argmax(Q)

print("\n==========================================")
print("BEST CONTENT USING UCB")
print("==========================================")

print(
    "Optimal Content:",
    contents[best_action]
)

print(
    "Estimated Engagement:",
    round(Q[best_action], 3)
)


# ============================================================
# RANDOM STRATEGY
# ============================================================

print("\n==========================================")
print("RANDOM STRATEGY")
print("==========================================")

random_reward = 0

for step in range(episodes * steps):

    action = random.randint(
        0,
        num_actions - 1
    )

    reward = environment_step(action)

    random_reward += reward


print(
    "Random Strategy Reward:",
    random_reward
)


# ============================================================
# EPSILON-GREEDY STRATEGY
# ============================================================

print("\n==========================================")
print("EPSILON-GREEDY STRATEGY")
print("==========================================")

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

epsilon_Q = np.zeros(num_actions)

epsilon_reward = 0

for step in range(episodes * steps):

    # Exploration or exploitation

    if random.random() < epsilon:

        action = random.randint(
            0,
            num_actions - 1
        )

    else:

        action = np.argmax(epsilon_Q)

    reward = environment_step(action)

    epsilon_Q[action] += 0.1 * (
        reward - epsilon_Q[action]
    )

    epsilon_reward += reward

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


print(
    "Epsilon-Greedy Reward:",
    epsilon_reward
)


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

print("\n==========================================")
print("STRATEGY COMPARISON")
print("==========================================")

print(
    "UCB Reward:",
    total_reward
)

print(
    "Random Reward:",
    random_reward
)

print(
    "Epsilon-Greedy Reward:",
    epsilon_reward
)


# ============================================================
# EFFECTIVENESS ANALYSIS
# ============================================================

print("\n==========================================")
print("EFFECTIVENESS ANALYSIS")
print("==========================================")

rewards = {
    "UCB": total_reward,
    "Random": random_reward,
    "Epsilon-Greedy": epsilon_reward
}

best_strategy = max(
    rewards,
    key=rewards.get
)

print(
    "Best Performing Strategy:",
    best_strategy
)

print(
    "Best Reward:",
    rewards[best_strategy]
)

print("\nUCB Streaming Content Recommendation Completed.")
