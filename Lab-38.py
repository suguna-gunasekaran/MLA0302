import random

print("META-LEARNING ADAPTIVE CONTROL SYSTEM")

conditions = int(input("Enter number of operating conditions: "))
episodes = int(input("Enter number of episodes: "))

# Meta-learned control parameters
control = 0.5
total_error = 0

for c in range(1, conditions + 1):
    target = float(input(f"\nEnter target value for Condition {c}: "))
    environment = float(input(f"Enter environment value for Condition {c}: "))

    print(f"\n--- Condition {c} ---")

    for e in range(1, episodes + 1):

        # Control action
        output = control * environment

        # Calculate error
        error = target - output

        # Performance feedback
        reward = -abs(error)

        # Meta-learning: adapt control parameter
        control += 0.1 * error

        # Keep parameter within valid range
        control = max(0, min(control, 2))

        total_error += abs(error)

        print(f"Episode {e}: Output={output:.2f}, "
              f"Error={error:.2f}, Control={control:.2f}")

# Performance evaluation
average_error = total_error / (conditions * episodes)

print("\n===== PERFORMANCE =====")
print(f"Final Control Parameter: {control:.2f}")
print(f"Average Error: {average_error:.2f}")

if average_error < 1:
    print("Performance: EXCELLENT")
elif average_error < 3:
    print("Performance: GOOD")
else:
    print("Performance: NEEDS IMPROVEMENT")
