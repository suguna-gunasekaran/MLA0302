# Policy Iteration for Delivery Drone

SIZE = 5

# Goal (Delivery Point)
GOAL = (4, 4)

# Obstacles
OBSTACLES = [(1, 2), (2, 2), (3, 1)]

# Discount Factor
gamma = 0.9

# Possible Actions
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Initialize Value Function
V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

# Initial Policy (Move Right)
policy = [["R" for _ in range(SIZE)] for _ in range(SIZE)]


def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE and (x, y) not in OBSTACLES


def reward(state):
    if state == GOAL:
        return 10
    return -1


stable = False

while not stable:

    # -------- Policy Evaluation --------
    for _ in range(20):
        newV = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

        for i in range(SIZE):
            for j in range(SIZE):

                if (i, j) == GOAL or (i, j) in OBSTACLES:
                    continue

                a = policy[i][j]
                dx, dy = actions[a]

                ni = i + dx
                nj = j + dy

                if not valid(ni, nj):
                    ni, nj = i, j

                newV[i][j] = reward((ni, nj)) + gamma * V[ni][nj]

        V = newV

    # -------- Policy Improvement --------
    stable = True

    for i in range(SIZE):
        for j in range(SIZE):

            if (i, j) == GOAL or (i, j) in OBSTACLES:
                continue

            old = policy[i][j]

            best_action = old
            best_value = -9999

            for a, (dx, dy) in actions.items():

                ni = i + dx
                nj = j + dy

                if not valid(ni, nj):
                    ni, nj = i, j

                value = reward((ni, nj)) + gamma * V[ni][nj]

                if value > best_value:
                    best_value = value
                    best_action = a

            policy[i][j] = best_action

            if old != best_action:
                stable = False


print("Optimal Value Function\n")

for row in V:
    for v in row:
        print(f"{v:6.2f}", end=" ")
    print()

print("\nOptimal Policy\n")

for i in range(SIZE):
    for j in range(SIZE):

        if (i, j) == GOAL:
            print(" G ", end=" ")

        elif (i, j) in OBSTACLES:
            print(" X ", end=" ")

        else:
            print(policy[i][j], end="  ")

    print()
