# ============================================================
# MOUNTAIN CAR USING POLICY GRADIENT
# TensorFlow / Keras + Gymnasium
# ============================================================

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# ---------------- INPUT ----------------

episodes = int(input("Enter number of training episodes: "))


# ---------------- ENVIRONMENT ----------------

env = gym.make("MountainCar-v0")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print("\nEnvironment: MountainCar-v0")
print("State size :", state_size)
print("Actions    :", action_size)


# ---------------- POLICY NETWORK ----------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(state_size,)),
    Dense(32, activation="relu"),
    Dense(action_size, activation="softmax")
])


optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

gamma = 0.99


# ============================================================
# TRAINING
# ============================================================

print("\n========== TRAINING ==========\n")

for episode in range(1, episodes + 1):

    state, info = env.reset()

    states = []
    actions = []
    rewards = []

    total_reward = 0

    for step in range(200):

        # Convert state to array
        state_input = np.reshape(
            state,
            [1, state_size]
        )

        # Get action probabilities
        probabilities = model(
            state_input,
            training=False
        ).numpy()[0]

        # Select action according to policy
        action = np.random.choice(
            action_size,
            p=probabilities
        )

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward

        state = next_state

        if terminated or truncated:
            break

    # --------------------------------------------------------
    # Calculate discounted rewards
    # --------------------------------------------------------

    discounted_rewards = []
    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G
        discounted_rewards.insert(0, G)

    discounted_rewards = np.array(
        discounted_rewards,
        dtype=np.float32
    )

    # Normalize rewards
    if len(discounted_rewards) > 1:
        discounted_rewards = (
            discounted_rewards -
            np.mean(discounted_rewards)
        ) / (
            np.std(discounted_rewards) + 1e-8
        )

    states = np.array(
        states,
        dtype=np.float32
    )

    actions = np.array(actions)

    # --------------------------------------------------------
    # Update policy
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        probabilities = model(
            states,
            training=True
        )

        # Probability of selected actions
        action_probabilities = tf.reduce_sum(
            probabilities *
            tf.one_hot(actions, action_size),
            axis=1
        )

        # Policy-gradient loss
        loss = -tf.reduce_mean(
            tf.math.log(
                action_probabilities + 1e-8
            ) * discounted_rewards
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if episode == 1 or episode % 10 == 0:

        print(
            "Episode:",
            episode,
            "| Steps:",
            len(rewards),
            "| Reward:",
            round(total_reward, 2)
        )


# ============================================================
# TEST TRAINED POLICY
# ============================================================

print("\n========== TESTING POLICY ==========\n")

test_episodes = 5

for episode in range(1, test_episodes + 1):

    state, info = env.reset()

    total_reward = 0

    for step in range(200):

        state_input = np.reshape(
            state,
            [1, state_size]
        )

        # Choose best action
        probabilities = model(
            state_input,
            training=False
        ).numpy()[0]

        action = np.argmax(probabilities)

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(
        "Test Episode:",
        episode,
        "| Steps:",
        step + 1,
        "| Reward:",
        total_reward
    )

env.close()

print("\nTraining and testing completed.")
