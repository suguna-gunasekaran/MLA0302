# ============================================================
# MONTE CARLO RL FRAMEWORK FOR CUSTOMER CHURN PREDICTION
# ============================================================

import numpy as np
import random

# ---------------- INPUT ----------------

episodes = int(input("Enter number of episodes: "))
steps = int(input("Enter number of steps per episode: "))

print("\nCustomer States:")
print("0 - Low Churn Risk")
print("1 - Medium Churn Risk")
print("2 - High Churn Risk")

# ---------------- ENVIRONMENT ----------------

states = {
    0: "Low Churn Risk",
    1: "Medium Churn Risk",
    2: "High Churn Risk"
}

actions = {
    0: "No Action",
    1: "Send Offer",
    2: "Send Retention Message"
}

num_states = 3
num_actions = 3


# ---------------- VALUE FUNCTION ----------------

# V[state] stores estimated value
V = np.zeros(num_states)

# Store returns for every state
returns = {
    0: [],
    1: [],
    2: []
}

gamma = 0.9


# ---------------- POLICY ----------------

# Fixed policy
# Each state has a preferred action

policy = {
    0: 0,   # Low risk -> No Action
    1: 1,   # Medium risk -> Send Offer
    2: 2    # High risk -> Retention Message
}


# ---------------- ENVIRONMENT STEP ----------------

def environment_step(state, action):

    # Reward based on action and churn risk

    if state == 0:

        # Low risk customer
        if action == 0:
            reward = 5
        else:
            reward = 2

    elif state == 1:

        # Medium risk customer
        if action == 1:
            reward = 8
        else:
            reward = 3

    else:

        # High risk customer
        if action == 2:
            reward = 10
        else:
            reward = -5

    # Generate next customer state
    next_state = random.randint(0, 2)

    return next_state, reward


# ============================================================
# MONTE CARLO POLICY EVALUATION
# ============================================================

print("\n==========================================")
print("MONTE CARLO CUSTOMER CHURN EVALUATION")
print("==========================================")

for episode in range(1, episodes + 1):

    episode_states = []
    episode_rewards = []

    # Initial customer state
    state = random.randint(0, 2)

    # --------------------------------------------------------
    # GENERATE EPISODE
    # --------------------------------------------------------

    for step in range(steps):

        # Follow fixed policy
        action = policy[state]

        # Environment
        next_state, reward = environment_step(
            state,
            action
        )

        # Store state and reward
        episode_states.append(state)
        episode_rewards.append(reward)

        # Move to next state
        state = next_state

    # --------------------------------------------------------
    # CALCULATE RETURNS
    # --------------------------------------------------------

    G = 0

    visited_states = set()

    for t in range(len(episode_states) - 1, -1, -1):

        G = gamma * G + episode_rewards[t]

        state = episode_states[t]

        # First-visit Monte Carlo
        if state not in visited_states:

            returns[state].append(G)

            V[state] = np.mean(
                returns[state]
            )

            visited_states.add(state)

    # Display training progress
    if episode == 1 or episode % max(1, episodes // 10) == 0:

        print(
            "Episode:",
            episode,
            "| Value Function:",
            np.round(V, 2)
        )


# ============================================================
# LEARNED VALUE FUNCTION
# ============================================================

print("\n==========================================")
print("ESTIMATED VALUE FUNCTION")
print("==========================================")

for state in range(num_states):

    print(
        states[state],
        "-> Value =",
        round(V[state], 2)
    )


# ============================================================
# POLICY
# ============================================================

print("\n==========================================")
print("EVALUATED CUSTOMER CHURN POLICY")
print("==========================================")

for state in range(num_states):

    action = policy[state]

    print(
        states[state],
        "->",
        actions[action]
    )


# ============================================================
# TEST THE POLICY
# ============================================================

print("\n==========================================")
print("TESTING CUSTOMER CHURN POLICY")
print("==========================================")

state = random.randint(0, 2)

total_reward = 0

for step in range(10):

    action = policy[state]

    next_state, reward = environment_step(
        state,
        action
    )

    total_reward += reward

    print(
        "Step:",
        step + 1,
        "| Customer State:",
        states[state],
        "| Action:",
        actions[action],
        "| Reward:",
        reward
    )

    state = next_state


# ============================================================
# FINAL ANALYSIS
# ============================================================

print("\n==========================================")
print("FINAL ANALYSIS")
print("==========================================")

best_state = np.argmax(V)

print(
    "Highest Value Customer State:",
    states[best_state]
)

print(
    "Estimated Value:",
    round(V[best_state], 2)
)

print(
    "Total Test Reward:",
    round(total_reward, 2)
)

print("\nMonte Carlo Customer Churn Policy Evaluation Completed.")
