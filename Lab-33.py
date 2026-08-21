import numpy as np
from sklearn.linear_model import LinearRegression

# -------- USER INPUT --------
n = int(input("Enter number of historical periods: "))

returns = []
print("Enter historical portfolio returns:")

for i in range(n):
    r = float(input(f"Period {i+1} return (%): "))
    returns.append(r)

years = int(input("Enter prediction period (years): "))

# -------- DATA PREPARATION --------
X = np.arange(1, n + 1).reshape(-1, 1)
y = np.array(returns)

# ML model
model = LinearRegression()
model.fit(X, y)

# Predict future return
future = np.array([[n + years]])
predicted_return = model.predict(future)[0]

# -------- PORTFOLIO INPUT --------
initial = float(input("Enter initial investment: "))
p = int(input("Enter number of portfolio strategies: "))

results = []

for i in range(p):
    name = input(f"Enter strategy {i+1} name: ")
    allocation = float(input(f"Enter allocation percentage for {name}: "))

    value = initial * (1 + predicted_return / 100) ** years
    value = value * (allocation / 100)

    results.append((name, value))

# -------- RESULTS --------
print("\n--- PREDICTED PORTFOLIO PERFORMANCE ---")

for name, value in results:
    print(name, ":", round(value, 2))

best = max(results, key=lambda x: x[1])

print("\nBest Strategy:", best[0])
print("Predicted Value:", round(best[1], 2))
