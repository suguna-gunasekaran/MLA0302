import random

print("MULTI-AGENT REINFORCEMENT LEARNING")

agents = int(input("Enter number of robots: "))
episodes = int(input("Enter number of episodes: "))
tasks = int(input("Enter number of tasks: "))

total_reward = 0
completed = 0

for ep in range(1, episodes + 1):
    print(f"\n--- Episode {ep} ---")
    episode_reward = 0

    for task in range(1, tasks + 1):
        # Robots choose actions
        actions = [random.choice(["Move", "Search", "Wait"])
                   for _ in range(agents)]

        # Cooperative reward
        if "Search" in actions:
            reward = 10
            completed += 1
        else:
            reward = -2

        # Coordination bonus
        if actions.count("Search") >= 2:
            reward += 5

        episode_reward += reward

        print(f"Task {task}: {actions}")
        print("Team Reward:", reward)

    total_reward += episode_reward
    print("Episode Reward:", episode_reward)

# Performance evaluation
total_tasks = episodes * tasks
success_rate = (completed / total_tasks) * 100

print("\n===== PERFORMANCE =====")
print("Total Team Reward:", total_reward)
print(f"Task Success Rate: {success_rate:.2f}%")

if success_rate >= 70:
    print("Performance: EXCELLENT COORDINATION")
elif success_rate >= 50:
    print("Performance: GOOD COORDINATION")
else:
    print("Performance: NEEDS IMPROVEMENT")
