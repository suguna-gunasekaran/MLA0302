# Policy Iteration for Delivery Drone

# Grid Size
SIZE = int(input("Enter Grid Size (e.g., 5): "))

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

# Policy Evaluation Iterations
eval_iterations = int(input("Enter Policy Evaluation Iterations: "))

# Actions
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Initialize Value Function
V = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

# Initial Policy
policy = [["R" for _ in range(SIZE)] for _ in range(SIZE)]

# Check Valid Position
def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE and (x, y) not in OBSTACLES

# Reward Function
def reward(state):
    if state == GOAL:
        return 10
    return -1

stable = False

while not stable:

    # Policy Evaluation
    for _ in range(eval_iterations):

        newV = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

        for i in range(SIZE):
            for j in range(SIZE):

                if (i, j) == GOAL or (i, j) in OBSTACLES:
                    continue

                action = policy[i][j]
                dx, dy = actions[action]

                ni = i + dx
                nj = j + dy

                if not valid(ni, nj):
                    ni, nj = i, j

                newV[i][j] = reward((ni, nj)) + gamma * V[ni][nj]

        V = newV

    # Policy Improvement
    stable = True

    for i in range(SIZE):
        for j in range(SIZE):

            if (i, j) == GOAL or (i, j) in OBSTACLES:
                continue

            old_action = policy[i][j]
            best_action = old_action
            best_value = float("-inf")

            for action, (dx, dy) in actions.items():

                ni = i + dx
                nj = j + dy

                if not valid(ni, nj):
                    ni, nj = i, j

                value = reward((ni, nj)) + gamma * V[ni][nj]

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[i][j] = best_action

            if old_action != best_action:
                stable = False

# Display Value Function
print("\nOptimal Value Function:\n")
for row in V:
    for value in row:
        print(f"{value:7.2f}", end=" ")
    print()

# Display Policy
print("\nOptimal Policy:\n")
for i in range(SIZE):
    for j in range(SIZE):
        if (i, j) == GOAL:
            print(" G ", end=" ")
        elif (i, j) in OBSTACLES:
            print(" X ", end=" ")
        else:
            print(policy[i][j], end="  ")
    print()
