import random
import math
import matplotlib.pyplot as plt

# -------- USER INPUT --------
W = int(input("Enter map width: "))
H = int(input("Enter map height: "))

sx = int(input("Enter start X: "))
sy = int(input("Enter start Y: "))

gx = int(input("Enter goal X: "))
gy = int(input("Enter goal Y: "))

n = int(input("Enter number of obstacles: "))

obstacles = []
for i in range(n):
    x = int(input(f"Obstacle {i+1} X: "))
    y = int(input(f"Obstacle {i+1} Y: "))
    w = int(input(f"Obstacle {i+1} Width: "))
    h = int(input(f"Obstacle {i+1} Height: "))
    obstacles.append((x, y, w, h))

iterations = int(input("Enter number of RRT iterations: "))

# -------- FUNCTIONS --------
def collision(x, y):
    for ox, oy, ow, oh in obstacles:
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            return True
    return False


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def valid_path(a, b):
    for i in range(11):
        t = i / 10
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])

        if collision(x, y):
            return False

    return True


# -------- RRT --------
tree = [(sx, sy)]
parent = {tree[0]: None}
step = 2

for _ in range(iterations):

    # Random point
    if random.random() < 0.1:
        point = (gx, gy)
    else:
        point = (random.uniform(0, W), random.uniform(0, H))

    # Find nearest node
    nearest = min(tree, key=lambda p: distance(p, point))

    # Move towards random point
    d = distance(nearest, point)

    if d == 0:
        continue

    new = (
        nearest[0] + step * (point[0] - nearest[0]) / d,
        nearest[1] + step * (point[1] - nearest[1]) / d
    )

    if 0 <= new[0] <= W and 0 <= new[1] <= H:
        if not collision(new[0], new[1]) and valid_path(nearest, new):
            tree.append(new)
            parent[new] = nearest

            # Goal reached
            if distance(new, (gx, gy)) < step:
                parent[(gx, gy)] = new
                tree.append((gx, gy))
                print("Goal reached!")
                break

# -------- FIND PATH --------
path = []

if (gx, gy) in parent:
    node = (gx, gy)

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()
    print("Collision-free path found!")
else:
    print("Path not found. Try increasing iterations.")

# -------- VISUALIZATION --------
plt.figure(figsize=(7, 7))

# Obstacles
for x, y, w, h in obstacles:
    plt.gca().add_patch(
        plt.Rectangle((x, y), w, h, color="black")
    )

# RRT tree
for node in tree:
    if parent[node] is not None:
        p = parent[node]
        plt.plot(
            [node[0], p[0]],
            [node[1], p[1]],
            "b-", linewidth=0.5
        )

# Path
if path:
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    plt.plot(px, py, "r-", linewidth=3, label="RRT Path")

plt.plot(sx, sy, "go", markersize=8, label="Start")
plt.plot(gx, gy, "ro", markersize=8, label="Goal")

plt.xlim(0, W)
plt.ylim(0, H)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("RRT Autonomous Exploration Robot")
plt.legend()
plt.grid()
plt.show()
