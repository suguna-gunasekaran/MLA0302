# Simulate an autonomous car navigating a simple road network with intersections. Design
# policies for the car to follow traffic rules and reach the destination safely. Implement these
# policies in Python and evaluate their effectiveness.

import random

# Grid Size
SIZE = int(input("Enter Grid Size: "))

# Start Position
start_x = int(input("Enter Start Row: "))
start_y = int(input("Enter Start Column: "))
START = (start_x, start_y)

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

# Possible Moves
moves = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Check Valid Position
def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE and (x, y) not in OBSTACLES

# ---------------- Random Policy ----------------
def random_policy():

    pos = START
    steps = 0

    print("\nRandom Policy Path:")

    while pos != GOAL and steps < 50:

        action = random.choice(list(moves.values()))

        nx = pos[0] + action[0]
        ny = pos[1] + action[1]

        if valid(nx, ny):
            pos = (nx, ny)

        print(pos)

        steps += 1

    if pos == GOAL:
        print("Goal Reached!")
    else:
        print("Goal Not Reached.")

    print("Steps:", steps)

# ---------------- Safe Policy ----------------
def safe_policy():

    pos = START
    steps = 0

    print("\nSafe Policy Path:")

    while pos != GOAL and steps < 100:

        x, y = pos

        if x < GOAL[0] and valid(x + 1, y):
            pos = (x + 1, y)

        elif y < GOAL[1] and valid(x, y + 1):
            pos = (x, y + 1)

        elif valid(x, y + 1):
            pos = (x, y + 1)

        elif valid(x + 1, y):
            pos = (x + 1, y)

        else:
            break

        print(pos)
        steps += 1

    if pos == GOAL:
        print("Goal Reached!")
    else:
        print("Goal Not Reached.")

    print("Steps:", steps)

# Main Program
print("\nAutonomous Car Navigation")
print("-------------------------")

random_policy()
safe_policy()
