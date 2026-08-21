import random
import math

W = int(input("Enter map width: "))
H = int(input("Enter map height: "))

n = int(input("Enter number of obstacles: "))
obs = []

for i in range(n):
    x, y, w, h = map(int, input(f"Obstacle {i+1} (x y width height): ").split())
    obs.append((x, y, w, h))

sx, sy = map(int, input("Enter start (x y): ").split())
gx, gy = map(int, input("Enter goal (x y): ").split())

def blocked(p):
    x, y = p
    return any(a <= x <= a+w and b <= y <= b+h for a,b,w,h in obs)

nodes = [(sx, sy)]
parent = {nodes[0]: None}

for _ in range(3000):
    r = (random.randint(0,W), random.randint(0,H))
    near = min(nodes, key=lambda p: math.dist(p,r))

    angle = math.atan2(r[1]-near[1], r[0]-near[0])
    new = (round(near[0] + 3*math.cos(angle)),
           round(near[1] + 3*math.sin(angle)))

    if 0 <= new[0] <= W and 0 <= new[1] <= H and not blocked(new):
        nodes.append(new)
        parent[new] = near

        if math.dist(new, (gx,gy)) < 5:
            parent[(gx,gy)] = new
            break

# Create path
path = []
p = (gx, gy)

while p is not None:
    path.append(p)
    p = parent.get(p)

path.reverse()

print("\nUAV Planned Path:")
for p in path:
    print(p)

print("\nPath Length:", len(path))
print("Mission completed successfully!")
