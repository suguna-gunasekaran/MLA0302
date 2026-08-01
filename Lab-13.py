import numpy as np

# States (0 to 4)
q = np.zeros((5, 2))   # Actions: Left, Right

alpha = 0.1
gamma = 0.9

for episode in range(100):
    state = 0

    while state < 4:

        action = np.argmax(q[state])

        next_state = state + 1

        reward = 10 if next_state == 4 else -1

        # Q-Learning Update
        q[state][action] = q[state][action] + alpha * (
            reward + gamma * np.max(q[next_state]) - q[state][action]
        )

        state = next_state

print("Q-Table")
print(q)

actions = ["Left", "Right"]

print("\nBest Action")
for i in range(5):
    print("State", i, ":", actions[np.argmax(q[i])])
