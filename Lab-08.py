import random

SIZE = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = [(1, 2), (2, 2), (3, 1)]

moves = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}


def valid(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE and (x, y) not in OBSTACLES


# ---------------- Random Policy ----------------

def random_policy():

    pos = START
    steps = 0

    print("\nRandom Policy")

    while pos != GOAL and steps < 50:

        action = random.choice(list(moves.values()))

        nx = pos[0] + action[0]
        ny = pos[1] + action[1]

        if valid(nx, ny):
            pos = (nx, ny)

        print(pos)

        steps += 1

    print("Steps:", steps)


# ---------------- Safe Policy ----------------

def safe_policy():

    pos = START
    steps = 0

    print("\nSafe Policy")

    while pos != GOAL:

        x, y = pos

        if x < GOAL[0] and valid(x + 1, y):
            pos = (x + 1, y)

        elif y < GOAL[1] and valid(x, y + 1):
            pos = (x, y + 1)

        elif valid(x, y + 1):
            pos = (x, y + 1)

        elif valid(x + 1, y):
            pos = (x + 1, y)

        print(pos)

        steps += 1

    print("Steps:", steps)


print("Autonomous Car Navigation")
print("-------------------------")

random_policy()

safe_policy()
