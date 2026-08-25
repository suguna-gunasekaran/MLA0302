import random

# MAXQ hierarchical task
class Agent:
    def __init__(self, name):
        self.name = name
        self.reward = 0

    def learn(self, subtask):
        # Simple learning policy
        if random.random() < 0.8:
            reward = 10
            action = "Successful"
        else:
            reward = -2
            action = "Failed"

        self.reward += reward
        print(f"{self.name} -> {subtask}: {action} (Reward = {reward})")


def maxq_task(agents, episodes, subtasks):
    total_reward = 0

    for episode in range(1, episodes + 1):
        print(f"\n--- Episode {episode} ---")

        episode_reward = 0

        # MAXQ hierarchy:
        # Root Task
        #   ├── Subtask 1
        #   ├── Subtask 2
        #   └── Subtask 3

        for i, agent in enumerate(agents):
            subtask = subtasks[i % len(subtasks)]
            before = agent.reward

            agent.learn(subtask)

            episode_reward += agent.reward - before

        total_reward += episode_reward

        print("Episode Reward:", episode_reward)

    return total_reward


# -------- USER INPUT --------
print("MAXQ Multi-Agent Cooperative Task")

n = int(input("Enter number of agents: "))
episodes = int(input("Enter number of episodes: "))

subtask_count = int(input("Enter number of subtasks: "))

subtasks = []
for i in range(subtask_count):
    subtasks.append(input(f"Enter name of subtask {i + 1}: "))

# Create agents
agents = [Agent(f"Agent-{i + 1}") for i in range(n)]

# Run MAXQ
total_reward = maxq_task(agents, episodes, subtasks)

# -------- PERFORMANCE --------
print("\n===== PERFORMANCE =====")

for agent in agents:
    print(f"{agent.name}: Total Reward = {agent.reward}")

print("\nOverall Reward:", total_reward)

if total_reward > 0:
    print("Overall Task Status: SUCCESS")
else:
    print("Overall Task Status: NEEDS IMPROVEMENT")
