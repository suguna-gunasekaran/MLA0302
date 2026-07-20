# Delivery Robot - Bellman Equation (Without Matplotlib)

SIZE = 5

# Delivery Point
GOAL = (4, 4)

# Obstacles
OBSTACLES = [(1, 2), (2, 2), (3, 1)]

gamma = 0.9
iterations = 20

# Initialize Value Function
V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

# Fixed Policy:
# Move RIGHT if possible, otherwise DOWN
def next_state(i, j):
    if j < SIZE - 1:
        return (i, j + 1)
    elif i < SIZE - 1:
        return (i + 1, j)
    else:
        return (i, j)

# Reward Function
def reward(state):
    if state == GOAL:
        return 10
    elif state in OBSTACLES:
        return -2
    else:
        return -1

# Bellman Equation
for k in range(iterations):

    newV = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

    for i in range(SIZE):
        for j in range(SIZE):

            if (i, j) == GOAL:
                continue

            ni, nj = next_state(i, j)

            newV[i][j] = reward((ni, nj)) + gamma * V[ni][nj]

    V = newV

# Display Value Function
print("Delivery Robot State-Value Function")
print("-----------------------------------")

for row in V:
    for value in row:
        print(f"{value:6.2f}", end=" ")
    print()

# Simple Grid Visualization
print("\nGrid Visualization")
print("------------------")

for i in range(SIZE):
    for j in range(SIZE):

        if (i, j) == GOAL:
            print(" G ", end=" ")

        elif (i, j) in OBSTACLES:
            print(" X ", end=" ")

        else:
            print(f"{int(V[i][j]):2}", end=" ")

    print()
