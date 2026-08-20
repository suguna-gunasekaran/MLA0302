#Train a virtual character to create engaging content (e.g., storytelling, interactive
#experiences) within a simulated virtual world using policy gradient methods. Implement the
#policy gradient algorithm in Python to optimize the character&#39;s behavior for maximum
#audience engagement.


# ============================================================
# POLICY GRADIENT FOR VIRTUAL CHARACTER CONTENT CREATION
# ============================================================

import numpy as np
import random

# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))
steps = int(input("Enter number of steps per episode: "))

learning_rate = float(
    input("Enter learning rate: ")
)

gamma = float(
    input("Enter discount factor (0 to 1): ")
)


# ---------------- VIRTUAL WORLD ----------------

print("\nVirtual Character Behaviors:")
print("0 - Tell a Story")
print("1 - Interactive Question")
print("2 - Funny Content")
print("3 - Adventure Experience")


actions = {
    0: "Tell a Story",
    1: "Interactive Question",
    2: "Funny Content",
    3: "Adventure Experience"
}

num_actions = 4


# ---------------- STATES ----------------

print("\nAudience States:")
print("0 - Low Engagement")
print("1 - Medium Engagement")
print("2 - High Engagement")

num_states = 3


# ============================================================
# ENVIRONMENT
# ============================================================

# Base engagement probability for each behavior

engagement_probability = {
    0: 0.65,
    1: 0.80,
    2: 0.75,
    3: 0.90
}


def environment_step(state, action):

    # Generate audience engagement

    if random.random() < engagement_probability[action]:

        # Positive engagement
        if state == 0:
            reward = 5
        elif state == 1:
            reward = 8
        else:
            reward = 10

    else:

        # Low engagement
        reward = 1

    # Determine next audience state
    if reward >= 8:
        next_state = 2

    elif reward >= 5:
        next_state = 1

    else:
        next_state = 0

    return next_state, reward


# ============================================================
# POLICY
# ============================================================

# Policy preferences for each state and action

policy_preferences = np.zeros(
    (num_states, num_actions)
)


# Softmax function
def softmax(values):

    values = values - np.max(values)

    probabilities = np.exp(values)

    return probabilities / np.sum(probabilities)


# ============================================================
# TRAINING
# ============================================================

print("\n==========================================")
print("TRAINING POLICY GRADIENT AGENT")
print("==========================================")

for episode in range(1, episodes + 1):

    state = 0

    episode_states = []
    episode_actions = []
    episode_rewards = []

    # --------------------------------------------------------
    # GENERATE EPISODE
    # --------------------------------------------------------

    for step in range(steps):

        # Calculate action probabilities
        probabilities = softmax(
            policy_preferences[state]
        )

        # Select action according to policy
        action = np.random.choice(
            num_actions,
            p=probabilities
        )

        # Environment
        next_state, reward = environment_step(
            state,
            action
        )

        # Store episode information
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        state = next_state


    # --------------------------------------------------------
    # CALCULATE RETURNS
    # --------------------------------------------------------

    returns = []

    G = 0

    for reward in reversed(
        episode_rewards
    ):

        G = reward + gamma * G

        returns.insert(
            0,
            G
        )


    # Normalize returns
    returns = np.array(returns)

    if np.std(returns) > 0:

        returns = (
            returns - np.mean(returns)
        ) / (
            np.std(returns) + 1e-8
        )


    # --------------------------------------------------------
    # POLICY GRADIENT UPDATE
    # --------------------------------------------------------

    for t in range(steps):

        state = episode_states[t]
        action = episode_actions[t]
        G = returns[t]

        probabilities = softmax(
            policy_preferences[state]
        )

        # Gradient of log policy
        gradient = -probabilities

        gradient[action] += 1

        # Update policy
        policy_preferences[state] += (
            learning_rate *
            G *
            gradient
        )


    # --------------------------------------------------------
    # TRAINING PROGRESS
    # --------------------------------------------------------

    if (
        episode == 1 or
        episode % max(
            1,
            episodes // 10
        ) == 0
    ):

        total_reward = sum(
            episode_rewards
        )

        print(
            "Episode:",
            episode,
            "| Total Reward:",
            total_reward
        )


# ============================================================
# LEARNED POLICY
# ============================================================

print("\n==========================================")
print("LEARNED VIRTUAL CHARACTER POLICY")
print("==========================================")

for state in range(num_states):

    probabilities = softmax(
        policy_preferences[state]
    )

    best_action = np.argmax(
        probabilities
    )

    print(
        "Audience State:",
        state,
        "| Best Behavior:",
        actions[best_action]
    )

    print(
        "Action Probabilities:",
        np.round(probabilities, 3)
    )


# ============================================================
# TEST TRAINED CHARACTER
# ============================================================

print("\n==========================================")
print("TESTING TRAINED CHARACTER")
print("==========================================")

state = 0
total_test_reward = 0

for step in range(10):

    probabilities = softmax(
        policy_preferences[state]
    )

    action = np.argmax(
        probabilities
    )

    next_state, reward = environment_step(
        state,
        action
    )

    total_test_reward += reward

    print(
        "Step:",
        step + 1,
        "| Audience State:",
        state,
        "| Behavior:",
        actions[action],
        "| Reward:",
        reward
    )

    state = next_state


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

print("\n==========================================")
print("PERFORMANCE ANALYSIS")
print("==========================================")

print(
    "Total Test Engagement Reward:",
    total_test_reward
)

average_reward = (
    total_test_reward / 10
)

print(
    "Average Engagement Reward:",
    round(average_reward, 2)
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

overall_probabilities = np.mean(
    [
        softmax(policy_preferences[state])
        for state in range(num_states)
    ],
    axis=0
)

best_behavior = np.argmax(
    overall_probabilities
)

print(
    "Most Preferred Content Behavior:",
    actions[best_behavior]
)

print(
    "Learned Preference:",
    round(
        overall_probabilities[best_behavior],
        3
    )
)

print(
    "\nPolicy Gradient Virtual Character Training Completed."
)
