# Value Iteration for Taxi Dispatching

SIZE = 5

# Pick-up Point (Goal)
GOAL = (4, 4)

# Obstacles
OBSTACLES = [(1, 2), (2, 2), (3, 1)]

gamma = 0.9
iterations = 20

actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Initialize Value Function
V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]


def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE and (x, y) not in OBSTACLES


def reward(state):
    if state == GOAL:
        return 10
    return -1


# ---------------- Value Iteration ----------------
for _ in range(iterations):

    newV = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

    for i in range(SIZE):
        for j in range(SIZE):

            if (i, j) == GOAL or (i, j) in OBSTACLES:
                continue

            best = -9999

            for action in actions.values():

                ni = i + action[0]
                nj = j + action[1]

                if not valid(ni, nj):
                    ni, nj = i, j

                value = reward((ni, nj)) + gamma * V[ni][nj]

                if value > best:
                    best = value

            newV[i][j] = best

    V = newV

# -------- Extract Optimal Policy --------
policy = [["" for _ in range(SIZE)] for _ in range(SIZE)]

for i in range(SIZE):
    for j in range(SIZE):

        if (i, j) == GOAL:
            policy[i][j] = "G"
            continue

        if (i, j) in OBSTACLES:
            policy[i][j] = "X"
            continue

        best_action = ""
        best_value = -9999

        for name, action in actions.items():

            ni = i + action[0]
            nj = j + action[1]

            if not valid(ni, nj):
                ni, nj = i, j

            value = reward((ni, nj)) + gamma * V[ni][nj]

            if value > best_value:
                best_value = value
                best_action = name

        policy[i][j] = best_action

# -------- Display Results --------
print("Optimal Value Function\n")

for row in V:
    for value in row:
        print(f"{value:6.2f}", end=" ")
    print()

print("\nOptimal Dispatch Policy\n")

for row in policy:
    print(" ".join(row))
