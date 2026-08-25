import random

print("POMDP ROBOT NAVIGATION")

size = int(input("Enter grid size: "))
episodes = int(input("Enter number of episodes: "))

goal = (size - 1, size - 1)

total_steps = 0
success = 0

for ep in range(1, episodes + 1):

    robot = (0, 0)
    steps = 0

    print(f"\n--- Episode {ep} ---")

    while robot != goal and steps < size * size * 2:

        # Partial observation
        sensor = random.choice(["Clear", "Obstacle", "Unknown"])

        # Navigation decision
        if sensor == "Obstacle":
            action = random.choice(["Right", "Down", "Wait"])
        else:
            # Move towards goal
            if robot[0] < goal[0]:
                action = "Right"
            elif robot[1] < goal[1]:
                action = "Down"
            else:
                action = "Wait"

        # Execute action with uncertainty
        x, y = robot

        if action == "Right" and x < size - 1:
            x += 1
        elif action == "Down" and y < size - 1:
            y += 1
        elif action == "Wait":
            pass

        # Random movement uncertainty
        if random.random() < 0.1:
            x, y = robot

        robot = (x, y)
        steps += 1

        print(f"Step {steps}: Sensor={sensor}, "
              f"Action={action}, Position={robot}")

    total_steps += steps

    if robot == goal:
        success += 1
        print("Result: Goal Reached")
    else:
        print("Result: Goal Not Reached")

# Performance evaluation
success_rate = (success / episodes) * 100
average_steps = total_steps / episodes

print("\n===== PERFORMANCE =====")
print("Successful Episodes:", success)
print(f"Success Rate: {success_rate:.2f}%")
print(f"Average Steps: {average_steps:.2f}")

if success_rate >= 80:
    print("Performance: EXCELLENT")
elif success_rate >= 50:
    print("Performance: GOOD")
else:
    print("Performance: NEEDS IMPROVEMENT")
