import numpy as np

# ============================================================
# BELLMAN OPTIMALITY EQUATION - ROBOT GRID NAVIGATION
# ============================================================

# ---------------- INPUT ----------------

rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

print("\nEnter start position:")
start_r = int(input("Start row (0 to {}): ".format(rows - 1)))
start_c = int(input("Start column (0 to {}): ".format(cols - 1)))

print("\nEnter goal position:")
goal_r = int(input("Goal row (0 to {}): ".format(rows - 1)))
goal_c = int(input("Goal column (0 to {}): ".format(cols - 1)))

obstacle_count = int(input("\nEnter number of obstacles: "))

obstacles = set()

for i in range(obstacle_count):
    print("\nObstacle", i + 1)
    r = int(input("Row: "))
    c = int(input("Column: "))
    obstacles.add((r, c))


# ---------------- PARAMETERS ----------------

gamma = 0.9
step_reward = -1
goal_reward = 100

# Actions: Up, Down, Left, Right
actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

# State-value table
V = np.zeros((rows, cols))


# ---------------- VALUE ITERATION ----------------

iterations = 0

while True:

    new_V = V.copy()

    maximum_change = 0

    for r in range(rows):
        for c in range(cols):

            # Skip obstacles
            if (r, c) in obstacles:
                continue

            # Goal state
            if (r, c) == (goal_r, goal_c):
                new_V[r][c] = goal_reward
                continue

            values = []

            for dr, dc in actions:

                nr = r + dr
                nc = c + dc

                # Check boundaries
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                # Cannot move into obstacle
                if (nr, nc) in obstacles:
                    continue

                # Reward
                if (nr, nc) == (goal_r, goal_c):
                    reward = goal_reward
                else:
                    reward = step_reward

                value = reward + gamma * V[nr][nc]

                values.append(value)

            # Bellman's Optimality Equation
            if values:
                new_V[r][c] = max(values)

            maximum_change = max(
                maximum_change,
                abs(new_V[r][c] - V[r][c])
            )

    V = new_V
    iterations += 1

    # Stop when values converge
    if maximum_change < 0.001:
        break

    # Safety limit
    if iterations >= 1000:
        break


# ---------------- DISPLAY VALUE FUNCTION ----------------

print("\n======================================")
print("OPTIMAL STATE-VALUE FUNCTION")
print("======================================")

for r in range(rows):

    for c in range(cols):

        if (r, c) in obstacles:
            print("  ###  ", end=" ")

        else:
            print("{:7.2f}".format(V[r][c]), end=" ")

    print()

print("\nIterations:", iterations)


# ---------------- FIND OPTIMAL PATH ----------------

print("\n======================================")
print("OPTIMAL PATH")
print("======================================")

current = (start_r, start_c)
path = [current]

visited = set()
visited.add(current)

while current != (goal_r, goal_c):

    r, c = current

    best_value = -float("inf")
    best_state = None

    for dr, dc in actions:

        nr = r + dr
        nc = c + dc

        # Check boundaries
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue

        # Skip obstacles
        if (nr, nc) in obstacles:
            continue

        # Avoid loops
        if (nr, nc) in visited:
            continue

        if V[nr][nc] > best_value:
            best_value = V[nr][nc]
            best_state = (nr, nc)

    # No valid path
    if best_state is None:
        print("No path to the goal!")
        break

    current = best_state
    path.append(current)
    visited.add(current)


# ---------------- PRINT PATH ----------------

if current == (goal_r, goal_c):

    for i, position in enumerate(path):

        if i == 0:
            print("Start  :", position)

        elif i == len(path) - 1:
            print("Goal   :", position)

        else:
            print("Step", i, " :", position)

    print("\nTotal steps:", len(path) - 1)


# ---------------- DISPLAY GRID ----------------

print("\n======================================")
print("GRID WITH OPTIMAL PATH")
print("======================================")

path_set = set(path)

for r in range(rows):

    for c in range(cols):

        if (r, c) == (start_r, start_c):
            print("  S  ", end=" ")

        elif (r, c) == (goal_r, goal_c):
            print("  G  ", end=" ")

        elif (r, c) in obstacles:
            print(" ### ", end=" ")

        elif (r, c) in path_set:
            print("  *  ", end=" ")

        else:
            print("  .  ", end=" ")

    print()

print("\nS = Start")
print("G = Goal")
print("* = Optimal Path")
print("# = Obstacle")
