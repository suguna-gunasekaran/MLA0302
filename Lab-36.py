#A logistics company aims to optimize its supply chain operations using model-based RL.
#Develop a data generation model that simulates order fulfillment processes, inventory flows,
#and transportation networks. Use Python to generate synthetic data and evaluate different
#supply chain management policies based on the simulated environment.


import random

# -------- USER INPUT --------
inventory = int(input("Enter initial inventory: "))
days = int(input("Enter number of days: "))
reorder_level = int(input("Enter reorder level: "))
order_qty = int(input("Enter order quantity: "))
transport_time = int(input("Enter transportation time (days): "))

# -------- DEMAND MODEL --------
def demand():
    return random.randint(5, 15)

# -------- POLICY 1: FIXED --------
def fixed_policy(stock):
    if stock <= reorder_level:
        return order_qty
    return 0

# -------- POLICY 2: AGGRESSIVE --------
def aggressive_policy(stock):
    if stock <= reorder_level + 5:
        return order_qty
    return 0

# -------- SIMULATION --------
def simulate(policy):

    stock = inventory
    pending = []
    reward = 0

    for day in range(1, days + 1):

        # Receive transported orders
        received = 0

        for item in pending[:]:
            if item[0] <= day:
                received += item[1]
                pending.remove(item)

        stock += received

        # Generate customer demand
        d = demand()

        # Fulfill order
        sold = min(stock, d)
        stock -= sold

        # Place order
        order = policy(stock)

        if order > 0:
            pending.append((day + transport_time, order))

        # Reward
        daily_reward = (sold * 10) - (stock * 2) - (order * 1)
        reward += daily_reward

        print(day, d, sold, order, stock, daily_reward)

    return reward


# -------- EVALUATION --------
print("\n--- FIXED POLICY ---")
r1 = simulate(fixed_policy)

print("\n--- AGGRESSIVE POLICY ---")
r2 = simulate(aggressive_policy)

print("\n--- SUPPLY CHAIN RESULTS ---")
print("Fixed Policy Reward     :", round(r1, 2))
print("Aggressive Policy Reward:", round(r2, 2))

if r1 > r2:
    print("Best Policy: Fixed Policy")
else:
    print("Best Policy: Aggressive Policy")
