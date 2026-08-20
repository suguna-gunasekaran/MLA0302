# ============================================================
# INVENTORY MANAGEMENT USING BELLMAN'S EQUATION
# FINDING OPTIMAL ORDERING POLICY
# ============================================================

import numpy as np

# ---------------- INPUT ----------------

states = int(input("Enter number of inventory states: "))
max_order = int(input("Enter maximum order quantity: "))

holding_cost = float(input("Enter holding cost per unit: "))
shortage_cost = float(input("Enter shortage cost per unit: "))
order_cost = float(input("Enter ordering cost per unit: "))

demand = int(input("Enter expected daily demand: "))

gamma = float(input("Enter discount factor (0 to 1): "))
iterations = int(input("Enter number of iterations: "))


# ---------------- ENVIRONMENT ----------------

print("\nInventory States:")

for i in range(states):

    print(
        i,
        "- Inventory:",
        i,
        "units"
    )


# ============================================================
# COST FUNCTION
# ============================================================

def calculate_cost(
    inventory,
    order
):

    # Inventory after ordering
    available = inventory + order

    # Inventory after demand
    remaining = available - demand

    # Holding cost
    if remaining > 0:

        holding = remaining * holding_cost
        shortage = 0

    else:

        holding = 0
        shortage = abs(remaining) * shortage_cost

    # Ordering cost
    ordering = order * order_cost

    total_cost = (
        holding +
        shortage +
        ordering
    )

    return total_cost


# ============================================================
# VALUE FUNCTION
# ============================================================

V = np.zeros(states)

# Optimal action for every state
policy = np.zeros(
    states,
    dtype=int
)


# ============================================================
# BELLMAN VALUE ITERATION
# ============================================================

print("\n==========================================")
print("BELLMAN VALUE ITERATION")
print("==========================================")

for iteration in range(1, iterations + 1):

    new_V = np.zeros(states)

    for inventory in range(states):

        costs = []

        # Try every possible order quantity
        for order in range(max_order + 1):

            immediate_cost = calculate_cost(
                inventory,
                order
            )

            # Next inventory state
            next_inventory = (
                inventory +
                order -
                demand
            )

            # Keep state inside range
            next_inventory = max(
                0,
                min(
                    states - 1,
                    next_inventory
                )
            )

            # Bellman's equation
            total_cost = (
                immediate_cost +
                gamma * V[next_inventory]
            )

            costs.append(total_cost)

        # Choose minimum cost action
        best_action = np.argmin(costs)

        new_V[inventory] = costs[best_action]

        policy[inventory] = best_action

    V = new_V


# ============================================================
# OPTIMAL POLICY
# ============================================================

print("\n==========================================")
print("OPTIMAL INVENTORY POLICY")
print("==========================================")

for inventory in range(states):

    print(
        "Inventory:",
        inventory,
        "-> Order:",
        policy[inventory],
        "units",
        "| Estimated Cost:",
        round(V[inventory], 2)
    )


# ============================================================
# VALUE FUNCTION
# ============================================================

print("\n==========================================")
print("VALUE FUNCTION")
print("==========================================")

for inventory in range(states):

    print(
        "Inventory:",
        inventory,
        "| Minimum Expected Cost:",
        round(V[inventory], 2)
    )


# ============================================================
# TEST OPTIMAL POLICY
# ============================================================

print("\n==========================================")
print("TESTING OPTIMAL POLICY")
print("==========================================")

inventory = states // 2

total_optimal_cost = 0

for day in range(10):

    order = policy[inventory]

    cost = calculate_cost(
        inventory,
        order
    )

    next_inventory = (
        inventory +
        order -
        demand
    )

    next_inventory = max(
        0,
        min(
            states - 1,
            next_inventory
        )
    )

    total_optimal_cost += cost

    print(
        "Day:",
        day + 1,
        "| Inventory:",
        inventory,
        "| Order:",
        order,
        "| Cost:",
        round(cost, 2),
        "| Next Inventory:",
        next_inventory
    )

    inventory = next_inventory


# ============================================================
# SIMPLE POLICY
# ============================================================

print("\n==========================================")
print("TESTING SIMPLE ORDERING POLICY")
print("==========================================")

inventory = states // 2

total_simple_cost = 0

for day in range(10):

    # Simple policy:
    # Order fixed quantity every day
    order = max_order // 2

    cost = calculate_cost(
        inventory,
        order
    )

    next_inventory = (
        inventory +
        order -
        demand
    )

    next_inventory = max(
        0,
        min(
            states - 1,
            next_inventory
        )
    )

    total_simple_cost += cost

    print(
        "Day:",
        day + 1,
        "| Inventory:",
        inventory,
        "| Order:",
        order,
        "| Cost:",
        round(cost, 2),
        "| Next Inventory:",
        next_inventory
    )

    inventory = next_inventory


# ============================================================
# COST COMPARISON
# ============================================================

print("\n==========================================")
print("COST COMPARISON")
print("==========================================")

print(
    "Optimal Policy Cost:",
    round(total_optimal_cost, 2)
)

print(
    "Simple Policy Cost:",
    round(total_simple_cost, 2)
)

cost_reduction = (
    total_simple_cost -
    total_optimal_cost
)

print(
    "Cost Reduction:",
    round(cost_reduction, 2)
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

if total_optimal_cost < total_simple_cost:

    print(
        "Bellman's Optimal Policy minimizes inventory cost."
    )

else:

    print(
        "Simple policy has lower cost for these parameters."
    )

print(
    "Minimum Starting-State Cost:",
    round(V[states // 2], 2)
)

print(
    "\nInventory Optimization using Bellman's Equation Completed."
)
