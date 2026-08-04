# A delivery robot operates in a warehouse with predefined delivery points. Using Bellman
# equations, compute the state-value function for navigating to each delivery point.
# Implement this in Python and visualize the value function for different policies.

# Delivery Robot - Bellman Equation

# Grid Size
SIZE = int(input("Enter Grid Size: "))

# Goal Position
goal_x = int(input("Enter Goal Row: "))
goal_y = int(input("Enter Goal Column: "))
GOAL = (goal_x, goal_y)

# Obstacles
num_obstacles = int(input("Enter Number of Obstacles: "))
OBSTACLES = []

for i in range(num_obstacles):
    x = int(input(f"Enter Obstacle {i+1} Row: "))
    y = int(input(f"Enter Obstacle {i+1} Column: "))
    OBSTACLES.append((x, y))

# Discount Factor
gamma = float(input("Enter Discount Factor (e.g., 0.9): "))

# Number of Bellman Iterations
iterations = int(input("Enter Number of Iterations: "))

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
print("\nDelivery Robot State-Value Function")
print("-----------------------------------")

for row in V:
    for value in row:
        print(f"{value:7.2f}", end=" ")
    print()

# Grid Visualization
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
