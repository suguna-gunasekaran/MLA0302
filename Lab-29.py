# ============================================================
# A2C - Autonomous Vehicle Racing
# Advantage Actor-Critic for Aggressive Driving Policy
# ============================================================

import numpy as np
import random

# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))
track_length = int(input("Enter track length (e.g. 100): "))
max_steps = int(input("Enter maximum steps per race (e.g. 150): "))

# ---------------- ENVIRONMENT ----------------

# Actions
# 0 = Brake
# 1 = Maintain Speed
# 2 = Accelerate

ACTIONS = 3

# State:
# 0 = Slow speed
# 1 = Medium speed
# 2 = High speed
# 3 = Very high speed
# 4 = Near finish
STATES = 5

# Actor weights
actor = np.random.randn(STATES, ACTIONS) * 0.01

# Critic weights
critic = np.random.randn(STATES) * 0.01

# Learning parameters
actor_lr = 0.01
critic_lr = 0.05
gamma = 0.95


# ---------------- SOFTMAX ----------------

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# ---------------- STATE FUNCTION ----------------

def get_state(position, speed):

    if position >= track_length - 10:
        return 4

    if speed < 3:
        return 0
    elif speed < 6:
        return 1
    elif speed < 9:
        return 2
    else:
        return 3


# ---------------- TRAINING ----------------

print("\n==============================")
print("A2C AUTONOMOUS RACING")
print("==============================")

for episode in range(1, episodes + 1):

    position = 0
    speed = 2.0

    states = []
    actions = []
    rewards = []

    total_reward = 0

    for step in range(max_steps):

        state = get_state(position, speed)

        # Actor chooses action
        probabilities = softmax(actor[state])

        action = np.random.choice(
            ACTIONS,
            p=probabilities
        )

        old_speed = speed

        # ---------------- ACTION EFFECT ----------------

        if action == 0:
            speed -= 1.5

        elif action == 1:
            speed += 0.1

        elif action == 2:
            speed += 1.5

        # Speed limits
        speed = max(0, min(speed, 12))

        # Vehicle moves according to speed
        position += speed

        # ---------------- REWARD ----------------

        # Higher speed gives positive reward
        reward = speed * 0.2

        # Aggressive driving bonus
        if action == 2 and speed > 6:
            reward += 1.0

        # Penalty for braking
        if action == 0:
            reward -= 0.3

        # Small time penalty
        reward -= 0.5

        # Prevent excessive speed
        if speed > 10:
            reward -= 2

        # Finish
        done = False

        if position >= track_length:
            reward += 50
            done = True

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward

        if done:
            break

    # ---------------- A2C UPDATE ----------------

    # Calculate discounted returns
    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    # Normalize returns
    returns = np.array(returns)

    if len(returns) > 1:
        returns = (
            returns - np.mean(returns)
        ) / (np.std(returns) + 1e-8)

    # Update Actor and Critic
    for i in range(len(states)):

        state = states[i]
        action = actions[i]
        G = returns[i]

        # Critic prediction
        value = critic[state]

        # Advantage
        advantage = G - value

        # ---------------- CRITIC UPDATE ----------------

        critic[state] += critic_lr * advantage

        # ---------------- ACTOR UPDATE ----------------

        probabilities = softmax(actor[state])

        for a in range(ACTIONS):

            if a == action:
                gradient = 1 - probabilities[a]
            else:
                gradient = -probabilities[a]

            actor[state, a] += (
                actor_lr * advantage * gradient
            )

    # ---------------- DISPLAY ----------------

    if episode == 1 or episode % max(1, episodes // 10) == 0:

        print(
            "Episode:",
            episode,
            "| Steps:",
            len(rewards),
            "| Reward:",
            round(total_reward, 2),
            "| Position:",
            round(position, 2),
            "| Speed:",
            round(speed, 2)
        )


# ============================================================
# TEST THE TRAINED AGENT
# ============================================================

print("\n==============================")
print("TESTING TRAINED RACING AGENT")
print("==============================")

position = 0
speed = 2.0
total_reward = 0

for step in range(max_steps):

    state = get_state(position, speed)

    # Choose best learned action
    probabilities = softmax(actor[state])
    action = np.argmax(probabilities)

    old_position = position

    # Apply action
    if action == 0:
        speed -= 1.5

    elif action == 1:
        speed += 0.1

    elif action == 2:
        speed += 1.5

    speed = max(0, min(speed, 12))

    position += speed

    # Reward
    reward = speed * 0.2
    reward -= 0.5

    if action == 2 and speed > 6:
        reward += 1

    if speed > 10:
        reward -= 2

    if position >= track_length:
        reward += 50

    total_reward += reward

    action_name = [
        "BRAKE",
        "MAINTAIN",
        "ACCELERATE"
    ][action]

    print(
        f"Step {step + 1:3d} | "
        f"Position: {position:7.2f} | "
        f"Speed: {speed:5.2f} | "
        f"Action: {action_name}"
    )

    if position >= track_length:
        break


# ---------------- FINAL RESULT ----------------

print("\n==============================")
print("RACE RESULT")
print("==============================")

if position >= track_length:
    print("Race Status : FINISHED")
    print("Lap Time    :", step + 1, "steps")
else:
    print("Race Status : NOT FINISHED")
    print("Position    :", round(position, 2))

print("Total Reward:", round(total_reward, 2))

print("\nLearned Policy:")
print("------------------------------")

for state in range(STATES):

    probabilities = softmax(actor[state])
    best_action = np.argmax(probabilities)

    names = ["BRAKE", "MAINTAIN", "ACCELERATE"]

    print(
        f"State {state}: "
        f"{names[best_action]} "
        f"(Probability = {probabilities[best_action]:.2f})"
    )
