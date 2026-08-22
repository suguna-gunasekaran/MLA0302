#A dynamic pricing platform aims to optimize its pricing strategy using model-based RL.
#Develop a predictive model that forecasts customer demand and price sensitivities based on
#historical sales data. Use Python to train the predictive model and implement a modelbased
#policy optimization algorithm to dynamically adjust prices in response to changing market
#conditions.


import numpy as np
from sklearn.linear_model import LinearRegression

# -------- USER INPUT --------
n = int(input("Enter number of historical records: "))

prices = []
demands = []

for i in range(n):
    p = float(input(f"Enter price {i+1}: "))
    d = float(input(f"Enter demand {i+1}: "))
    prices.append(p)
    demands.append(d)

cost = float(input("Enter product cost: "))
min_price = float(input("Enter minimum price: "))
max_price = float(input("Enter maximum price: "))
step = float(input("Enter price step: "))

# -------- TRAIN DEMAND MODEL --------
X = np.array(prices).reshape(-1, 1)
y = np.array(demands)

model = LinearRegression()
model.fit(X, y)

# -------- MODEL-BASED POLICY OPTIMIZATION --------
best_price = 0
best_profit = -1

price = min_price

while price <= max_price:

    predicted_demand = model.predict([[price]])[0]

    # Demand cannot be negative
    predicted_demand = max(0, predicted_demand)

    profit = (price - cost) * predicted_demand

    if profit > best_profit:
        best_profit = profit
        best_price = price

    price += step

# -------- OUTPUT --------
predicted_demand = max(0, model.predict([[best_price]])[0])

print("\n--- DYNAMIC PRICING RESULT ---")
print("Optimal Price       :", round(best_price, 2))
print("Predicted Demand    :", round(predicted_demand, 2))
print("Predicted Profit    :", round(best_profit, 2))
