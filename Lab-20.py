#Implement an epsilon-greedy strategy to optimize content recommendations on an online
#learning platform. Write a Python script to simulate and analyze its performance over
#multiple runs.

# ============================================================
# RL FRAMEWORK FOR ONLINE LEARNING CONTENT RECOMMENDATION
# ============================================================

import numpy as np
import random

# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))
steps = int(input("Enter number of steps per episode: "))
runs = int(input("Enter number of simulation runs: "))

print("\nContent Options:")
print("0 - Python")
print("1 - Machine Learning")
print("2 - Data Structures")
print("3 - Web Development")


# ---------------- ENVIRONMENT ----------------

contents = {
    0: "Python",
    1: "Machine Learning",
    2: "Data Structures",
    3: "Web Development"
}

num_actions = 4

# Probability of receiving positive feedback
# for each content
engagement_probability = {
    0: 0.70,
    1: 0.85,
    2: 0.60,
    3: 0.45
}


# ---------------- VALUE FUNCTION ----------------

# Q[action] stores estimated value of each content
Q = np.zeros(num_actions)

learning_rate = 0.1

# Discount factor
gamma = 0.9

# Epsilon-greedy parameters
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995


# ---------------- ENVIRONMENT STEP ----------------

def environment_step(action):

    # Simulate learner engagement

    if random.random() < engagement_probability[action]:

        # Positive engagement
        reward = 1

    else:

        # No engagement
        reward = 0

    return reward


# ============================================================
# TRAINING
# ============================================================

print("\n==========================================")
print("TRAINING EPSILON-GREEDY RECOMMENDATION AGENT")
print("==========================================")

run_rewards = []

for run in range(1, runs + 1):

    # Reset Q-table for every run
    Q = np.zeros(num_actions)

    epsilon = 1.0

    total_run_reward = 0

    for episode in range(1, episodes + 1):

        total_reward = 0

        for step in range(steps):

            # ------------------------------------------------
            # EPSILON-GREEDY POLICY
            # ------------------------------------------------

            if random.random() < epsilon:

                # Explore
                action = random.randint(
                    0,
                    num_actions - 1
                )

            else:

                # Exploit
                action = np.argmax(Q)

            # ------------------------------------------------
            # ENVIRONMENT
            # ------------------------------------------------

            reward = environment_step(action)

            # ------------------------------------------------
            # Q VALUE UPDATE
            # ------------------------------------------------

            Q[action] += learning_rate * (
                reward - Q[action]
            )

            total_reward += reward

        # Reduce exploration
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        total_run_reward += total_reward

        # Display training progress
        if run == 1 and (
            episode == 1 or
            episode % max(1, episodes // 10) == 0
        ):

            print(
                "Episode:",
                episode,
                "| Reward:",
                total_reward,
                "| Epsilon:",
                round(epsilon, 3)
            )

    run_rewards.append(total_run_reward)


# ============================================================
# LEARNED VALUE FUNCTION
# ============================================================

print("\n==========================================")
print("LEARNED CONTENT VALUES")
print("==========================================")

for action in range(num_actions):

    print(
        contents[action],
        "-> Estimated Value =",
        round(Q[action], 3)
    )


# ============================================================
# LEARNED POLICY
# ============================================================

print("\n==========================================")
print("LEARNED RECOMMENDATION POLICY")
print("==========================================")

best_action = np.argmax(Q)

print(
    "Best Content Recommendation:",
    contents[best_action]
)

print(
    "Estimated Engagement Value:",
    round(Q[best_action], 3)
)


# ============================================================
# MULTIPLE RUN ANALYSIS
# ============================================================

print("\n==========================================")
print("MULTIPLE RUN PERFORMANCE")
print("==========================================")

for i in range(runs):

    print(
        "Run:",
        i + 1,
        "| Total Reward:",
        round(run_rewards[i], 2)
    )

average_reward = np.mean(run_rewards)
maximum_reward = np.max(run_rewards)
minimum_reward = np.min(run_rewards)


# ============================================================
# FINAL ANALYSIS
# ============================================================

print("\n==========================================")
print("FINAL ANALYSIS")
print("==========================================")

print(
    "Average Reward:",
    round(average_reward, 2)
)

print(
    "Maximum Reward:",
    round(maximum_reward, 2)
)

print(
    "Minimum Reward:",
    round(minimum_reward, 2)
)

print(
    "Optimal Content:",
    contents[best_action]
)

print("\nEpsilon-Greedy Content Recommendation Completed.")
