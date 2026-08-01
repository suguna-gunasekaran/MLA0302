import numpy as np

# Rooms: A, B, C, D
q = np.zeros((4, 2))  # 2 actions: Clean, Move

alpha = 0.1
gamma = 0.9

for episode in range(100):
    for state in range(3):

        action = np.argmax(q[state])

        reward = 10 if state == 2 else -1

        next_state = state + 1
        next_action = np.argmax(q[next_state])

        # SARSA Update
        q[state][action] = q[state][action] + alpha * (
            reward + gamma * q[next_state][next_action] - q[state][action]
        )

print("Q-Table")
print(q)

actions = ["Clean", "Move"]

print("\nBest Action")
for i in range(4):
    print("Room", i + 1, ":", actions[np.argmax(q[i])])
