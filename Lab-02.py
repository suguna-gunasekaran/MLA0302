# Policy Evaluation for Warehouse Robot

SIZE = 5

# Rewards
ITEMS = [(1, 1), (3, 2)]
GOAL = (4, 4)
OBSTACLES = [(2, 2), (3, 1)]

# Discount Factor
gamma = 0.9

# Number of Iterations
iterations = 15

# Initialize Value Function
V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]


# Fixed Policy:
# Move RIGHT if possible,
# otherwise move DOWN.
def next_state(i, j):

    if j < SIZE - 1:
        return (i, j + 1)

    elif i < SIZE - 1:
        return (i + 1, j)

    else:
        return (i, j)


# Reward Function
def reward(state):

    if state in ITEMS:
        return 2

    elif state == GOAL:
        return 5

    elif state in OBSTACLES:
        return -2

    else:
        return 0


# Policy Evaluation
for k in range(iterations):

    new_V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

    for i in range(SIZE):
        for j in range(SIZE):

            s_next = next_state(i, j)

            r = reward(s_next)

            new_V[i][j] = r + gamma * V[s_next[0]][s_next[1]]

    V = new_V


# Display Final Value Function
print("Warehouse Robot Policy Evaluation")
print("---------------------------------\n")

for row in V:
    for value in row:
        print(f"{value:6.2f}", end=" ")
    print()
