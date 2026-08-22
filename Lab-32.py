#A retail company aims to optimize its inventory management strategy using model-based RL.
#Develop a data generation model that simulates customer demand patterns and inventory
#dynamics. Use Python to generate synthetic data and evaluate different inventory
#management policies based on the simulated environment.


import random

# -------- USER INPUT --------
initial_inventory = int(input("Enter initial inventory: "))
days = int(input("Enter number of days: "))
reorder_level = int(input("Enter reorder level: "))
order_quantity = int(input("Enter order quantity: "))

# Demand model
def demand():
    return random.randint(5, 15)

# Policies
def fixed_policy(inventory):
    if inventory <= reorder_level:
        return order_quantity
    return 0

def aggressive_policy(inventory):
    if inventory <= reorder_level + 5:
        return order_quantity
    return 0

# -------- SIMULATION --------
def simulate(policy):
    inventory = initial_inventory
    total_reward = 0

    for day in range(1, days + 1):

        d = demand()
        order = policy(inventory)

        inventory += order
        sold = min(inventory, d)
        inventory -= sold

        # Reward = sales - holding cost - ordering cost
        reward = sold * 10 - inventory * 2 - order * 1
        total_reward += reward

        print(day, d, order, inventory, reward)

    return total_reward

# -------- POLICY EVALUATION --------
print("\n--- Fixed Reorder Policy ---")
reward1 = simulate(fixed_policy)

print("\n--- Aggressive Reorder Policy ---")
reward2 = simulate(aggressive_policy)

print("\n--- RESULTS ---")
print("Fixed Policy Reward     :", round(reward1, 2))
print("Aggressive Policy Reward:", round(reward2, 2))

if reward1 > reward2:
    print("Best Policy: Fixed Reorder Policy")
else:
    print("Best Policy: Aggressive Reorder Policy")
