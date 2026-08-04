# Implement dynamic programming methods to solve a gridworld navigation problem where
# the agent must reach the goal with the least number of steps while avoiding obstacles. Use
# Python to simulate the environment and policy iteration.

import numpy as np

# User Inputs
states = int(input("Enter Number of States: "))

grid = []
print("Enter Grid Values (0 = Empty, 1 = Goal)")
for i in range(states):
    value = int(input(f"State {i}: "))
    grid.append(value)

gamma = float(input("Enter Discount Factor (e.g., 0.9): "))
iterations = int(input("Enter Number of Policy Iterations: "))

# Initial Policy (Move Right)
policy = ["Right"] * states

# Initialize State Values
state_values = np.zeros(states)

# Policy Iteration
for k in range(iterations):

    for state in range(states - 2, -1, -1):

        if grid[state + 1] == 1:
            reward = 10
        else:
            reward = -1

        state_values[state] = reward + gamma * state_values[state + 1]

# Display State Values
print("\nState Values:")
print(state_values)

# Display Policy
print("\nOptimal Policy:")
for i in range(states):
    print(f"State {i}: {policy[i]}")
