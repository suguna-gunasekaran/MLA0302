#Simulate an RL framework to optimize a manufacturing process, where actions represent
#different machine settings and rewards are based on product quality. Implement the
#environment, policy, and value function in Python.


# ============================================================
# RL FRAMEWORK FOR MANUFACTURING PROCESS OPTIMIZATION
# ============================================================

import numpy as np
import random

# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))
steps = int(input("Enter number of steps per episode: "))

print("\nMachine Settings:")
print("0 - Low Temperature")
print("1 - Medium Temperature")
print("2 - High Temperature")

print("\nEnter desired product quality (0 to 100):")
target_quality = float(input("Target Quality: "))


# ---------------- ENVIRONMENT ----------------

# Actions represent machine settings
actions = {
    0: "Low Temperature",
    1: "Medium Temperature",
    2: "High Temperature"
}

# Expected quality for each machine setting
quality_output = {
    0: 60,
    1: 85,
    2: 70
}

# Number of states
# 0 = Low quality
# 1 = Medium quality
# 2 = High quality
num_states = 3
num_actions = 3


# ---------------- VALUE FUNCTION ----------------

# V[state] stores the estimated value of each state
V = np.zeros(num_states)

# Q-table used by the policy
Q = np.zeros((num_states, num_actions))

learning_rate = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995


# ---------------- STATE FUNCTION ----------------

def get_state(quality):

    if quality < 70:
        return 0

    elif quality < 85:
        return 1

    else:
        return 2


# ---------------- ENVIRONMENT STEP ----------------

def environment_step(action):

    # Base quality
    quality = quality_output[action]

    # Add small manufacturing variation
    variation = random.uniform(-5, 5)

    quality += variation

    # Keep quality between 0 and 100
    quality = max(0, min(100, quality))

    # Reward based on closeness to target quality
    reward = 100 - abs(target_quality - quality)

    return quality, reward


# ============================================================
# TRAINING
# ============================================================

print("\n==========================================")
print("TRAINING RL MANUFACTURING AGENT")
print("==========================================")

for episode in range(1, episodes + 1):

    # Initial state
    quality = 50
    state = get_state(quality)

    total_reward = 0

    for step in range(steps):

        # ----------------------------------------------------
        # POLICY
        # Epsilon-Greedy action selection
        # ----------------------------------------------------

        if random.random() < epsilon:

            action = random.randint(
                0,
                num_actions - 1
            )

        else:

            action = np.argmax(Q[state])

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        new_quality, reward = environment_step(action)

        next_state = get_state(new_quality)

        # ----------------------------------------------------
        # VALUE FUNCTION / Q UPDATE
        # ----------------------------------------------------

        best_next_value = np.max(
            Q[next_state]
        )

        target = (
            reward +
            gamma * best_next_value
        )

        Q[state, action] += learning_rate * (
            target - Q[state, action]
        )

        # Update state
        state = next_state
        quality = new_quality

        total_reward += reward

    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Display training progress
    if episode == 1 or episode % max(1, episodes // 10) == 0:

        print(
            "Episode:",
            episode,
            "| Total Reward:",
            round(total_reward, 2),
            "| Epsilon:",
            round(epsilon, 3)
        )


# ============================================================
# VALUE FUNCTION
# ============================================================

print("\n==========================================")
print("LEARNED VALUE FUNCTION")
print("==========================================")

for state in range(num_states):

    print(
        "State",
        state,
        "Value =",
        round(np.max(Q[state]), 2)
    )


# ============================================================
# LEARNED POLICY
# ============================================================

print("\n==========================================")
print("LEARNED POLICY")
print("==========================================")

state_names = [
    "Low Quality",
    "Medium Quality",
    "High Quality"
]

for state in range(num_states):

    best_action = np.argmax(Q[state])

    print(
        state_names[state],
        "->",
        actions[best_action]
    )


# ============================================================
# TEST THE TRAINED AGENT
# ============================================================

print("\n==========================================")
print("TESTING OPTIMAL MACHINE SETTING")
print("==========================================")

quality = 50
state = get_state(quality)

total_reward = 0

for step in range(10):

    # Select best action
    action = np.argmax(Q[state])

    # Perform action
    quality, reward = environment_step(action)

    state = get_state(quality)

    total_reward += reward

    print(
        "Step:",
        step + 1,
        "| Setting:",
        actions[action],
        "| Quality:",
        round(quality, 2),
        "| Reward:",
        round(reward, 2)
    )


# ============================================================
# FINAL RESULT
# ============================================================

best_action = np.argmax(
    np.mean(Q, axis=0)
)

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print(
    "Optimal Machine Setting:",
    actions[best_action]
)

print(
    "Average Estimated Value:",
    round(np.mean(Q[:, best_action]), 2)
)

print(
    "Total Test Reward:",
    round(total_reward, 2)
)

print("\nRL Manufacturing Optimization Completed.")
