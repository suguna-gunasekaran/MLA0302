import numpy as np

# Grid (0 = Empty, 1 = Goal)
grid = [0, 0, 0, 1]

# Initial Policy (Move Right)
policy = ["Right"] * 4

value = np.zeros(4)
gamma = 0.9

# Policy Iteration
for i in range(10):
    for state in range(2, -1, -1):
        reward = 10 if grid[state + 1] == 1 else -1
        value[state] = reward + gamma * value[state + 1]

print("State Values:")
print(value)

print("\nOptimal Policy:")
for i in range(4):
    print("State", i, ":", policy[i])
