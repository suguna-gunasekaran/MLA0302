import random

# ---------------- INPUT ----------------

SIZE = int(input("Enter Grid Size: "))

print("Enter Robot Start Position")
sx = int(input("Row: "))
sy = int(input("Column: "))
start = (sx, sy)

# Dirt Locations
dirt = set()
n = int(input("Enter Number of Dirt Cells: "))
print("Enter Dirt Cell Coordinates:")
for i in range(n):
    x, y = map(int, input().split())
    dirt.add((x, y))

# Obstacle Locations
obstacles = set()
m = int(input("Enter Number of Obstacle Cells: "))
print("Enter Obstacle Cell Coordinates:")
for i in range(m):
    x, y = map(int, input().split())
    obstacles.add((x, y))

# Possible Moves
moves = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Check valid position
def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE


# ---------------- Random Policy ----------------

def random_policy():
    position = start
    dirt_left = set(dirt)
    reward = 0
    steps = 0

    print("\n===== RANDOM POLICY =====")

    while dirt_left and steps < 50:

        action = random.choice(list(moves.keys()))
        dx, dy = moves[action]

        nx = position[0] + dx
        ny = position[1] + dy

        if valid(nx, ny):
            position = (nx, ny)

            if position in obstacles:
                reward -= 1
                print(position, "Obstacle (-1)")

            elif position in dirt_left:
                reward += 1
                dirt_left.remove(position)
                print(position, "Dirt Cleaned (+1)")

            else:
                print(position, "Empty Cell")

        steps += 1

    print("Total Reward =", reward)


# ---------------- Greedy Policy ----------------

def greedy_policy():

    position = start
    dirt_left = set(dirt)
    reward = 0

    print("\n===== GREEDY POLICY =====")

    while dirt_left:

        # Find nearest dirt
        target = min(
            dirt_left,
            key=lambda d: abs(position[0] - d[0]) + abs(position[1] - d[1])
        )

        # Move Row
        if position[0] < target[0]:
            next_pos = (position[0] + 1, position[1])

        elif position[0] > target[0]:
            next_pos = (position[0] - 1, position[1])

        # Move Column
        elif position[1] < target[1]:
            next_pos = (position[0], position[1] + 1)

        elif position[1] > target[1]:
            next_pos = (position[0], position[1] - 1)

        else:
            next_pos = position

        position = next_pos

        if position in obstacles:
            reward -= 1
            print(position, "Obstacle (-1)")

        elif position in dirt_left:
            reward += 1
            dirt_left.remove(position)
            print(position, "Dirt Cleaned (+1)")

        else:
            print(position, "Moving")

    print("Total Reward =", reward)


# ---------------- Main ----------------

print("\nAutonomous Cleaning Robot using MDP")

print("\nGrid Size:", SIZE, "x", SIZE)
print("Robot Start Position:", start)
print("Dirt Cells:", dirt)
print("Obstacle Cells:", obstacles)

random_policy()
greedy_policy()
