# Develop an AI agent to play a real-time strategy game (e.g., Age of Empires) using ActorCritic
# methods. Implement the actor and critic networks in Python and train the agent to build
# structures, gather resources, and engage in strategic combat.

import numpy as np

# User Input
resources = int(input("Enter resources: "))
army = int(input("Enter army units: "))
enemy = int(input("Enter enemy units: "))

# Actor probabilities
actor = np.ones(4) / 4

actions = ["Gather", "Build", "Train Army", "Attack"]

# Training
for episode in range(100):

    action = np.random.choice(4, p=actor)

    if action == 0:              # Gather
        resources += 10
        reward = 5

    elif action == 1 and resources >= 20:   # Build
        resources -= 20
        reward = 10

    elif action == 2 and resources >= 10:   # Train
        resources -= 10
        army += 1
        reward = 8

    elif action == 3 and army > 0 and enemy > 0:  # Attack
        army -= 1
        enemy -= 1
        reward = 20

    else:
        reward = -5

    # Actor-Critic learning
    critic = reward
    advantage = reward - critic

    if reward > 0:
        actor[action] += 0.01
    else:
        actor[action] -= 0.01

    actor = np.maximum(actor, 0.01)
    actor = actor / sum(actor)

# Final decision
action = np.argmax(actor)

print("\n--- RTS AI RESULT ---")
print("Action   :", actions[action])
print("Resources:", resources)
print("Army     :", army)
print("Enemies  :", enemy)

if enemy == 0:
    print("Result   : Enemy defeated!")
