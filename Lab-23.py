# ============================================================
# GRIDWORLD: POLICY AND VALUE FUNCTION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------- INPUT ----------------

grid_size = int(input("Enter grid size: "))
episodes = int(input("Enter number of evaluation episodes: "))
gamma = float(input("Enter discount factor (0 to 1): "))

print("\nActions:")
print("0 - Up")
print("1 - Down")
print("2 - Left")
print("3 - Right")


# ---------------- ENVIRONMENT ----------------

actions = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right"
}

num_actions = 4

start = (0, 0)
goal = (grid_size - 1, grid_size - 1)


# ============================================================
# MOVE FUNCTION
# ============================================================

def move(state, action):

    row, col = state

    if action == 0:       # Up
        row -= 1

    elif action == 1:     # Down
        row += 1

    elif action == 2:     # Left
        col -= 1

    elif action == 3:     # Right
        col += 1

    # Keep agent inside grid
    row = max(0, min(grid_size - 1, row))
    col = max(0, min(grid_size - 1, col))

    next_state = (row, col)

    # Goal reward
    if next_state == goal:
        reward = 10

    else:
        reward = -1

    return next_state, reward


# ============================================================
# POLICY 1
# MOVE TOWARDS GOAL
# ============================================================

def good_policy(state):

    row, col = state
    goal_row, goal_col = goal

    # Move down if possible
    if row < goal_row:
        return 1

    # Otherwise move right
    elif col < goal_col:
        return 3

    return 0


# ============================================================
# POLICY 2
# LESS EFFICIENT POLICY
# ============================================================

def bad_policy(state):

    row, col = state

    # Move right first
    if col < grid_size - 1:
        return 3

    # Then move down
    elif row < grid_size - 1:
        return 1

    return 0


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(policy):

    V = np.zeros(
        (grid_size, grid_size)
    )

    for episode in range(episodes):

        state = start

        visited = []

        rewards = []

        # Generate episode
        for step in range(
            grid_size * grid_size * 2
        ):

            if state == goal:
                break

            action = policy(state)

            next_state, reward = move(
                state,
                action
            )

            visited.append(state)
            rewards.append(reward)

            state = next_state

        # Calculate return
        G = 0

        for t in range(
            len(visited) - 1,
            -1,
            -1
        ):

            G = rewards[t] + gamma * G

            row, col = visited[t]

            V[row, col] = G

    return V


# ============================================================
# EVALUATE BOTH POLICIES
# ============================================================

print("\n==========================================")
print("EVALUATING POLICY 1")
print("==========================================")

V_good = evaluate_policy(
    good_policy
)

print(
    np.round(V_good, 2)
)


print("\n==========================================")
print("EVALUATING POLICY 2")
print("==========================================")

V_bad = evaluate_policy(
    bad_policy
)

print(
    np.round(V_bad, 2)
)


# ============================================================
# DISPLAY POLICY ACTIONS
# ============================================================

print("\n==========================================")
print("POLICY 1")
print("==========================================")

for row in range(grid_size):

    for col in range(grid_size):

        state = (row, col)

        if state == goal:

            print(" G ", end=" ")

        else:

            action = good_policy(state)

            print(
                actions[action][0],
                end="   "
            )

    print()


print("\n==========================================")
print("POLICY 2")
print("==========================================")

for row in range(grid_size):

    for col in range(grid_size):

        state = (row, col)

        if state == goal:

            print(" G ", end=" ")

        else:

            action = bad_policy(state)

            print(
                actions[action][0],
                end="   "
            )

    print()


# ============================================================
# VALUE COMPARISON
# ============================================================

print("\n==========================================")
print("VALUE FUNCTION COMPARISON")
print("==========================================")

print(
    "Policy 1 Value at Start:",
    round(V_good[start], 2)
)

print(
    "Policy 2 Value at Start:",
    round(V_bad[start], 2)
)


if V_good[start] > V_bad[start]:

    print(
        "\nPolicy 1 provides a higher expected return."
    )

else:

    print(
        "\nPolicy 2 provides a higher expected return."
    )


# ============================================================
# VISUALIZATION - POLICY 1
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    V_good,
    cmap="viridis"
)

plt.colorbar(
    label="State Value"
)

plt.title(
    "Value Function - Policy 1"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# VISUALIZATION - POLICY 2
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    V_bad,
    cmap="viridis"
)

plt.colorbar(
    label="State Value"
)

plt.title(
    "Value Function - Policy 2"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# DIFFERENCE BETWEEN POLICIES
# ============================================================

difference = V_good - V_bad

plt.figure(figsize=(6, 5))

plt.imshow(
    difference,
    cmap="coolwarm"
)

plt.colorbar(
    label="Value Difference"
)

plt.title(
    "Difference Between Value Functions"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print(
    "Policy 1 Start-State Value:",
    round(V_good[start], 2)
)

print(
    "Policy 2 Start-State Value:",
    round(V_bad[start], 2)
)

print(
    "Goal State Value:",
    round(V_good[goal], 2)
)

print(
    "\nPolicy and Value Function Analysis Completed."
)
