import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = np.array([300, 400, 500])
y = np.array([0.616, 0.525, 0.457])

# Number of data points
n = len(x)

# Formulate the design matrix for a quadratic fit
X = np.vstack((x**2, x, np.ones(n))).T
Y = y.reshape(-1, 1)

# Solve for the coefficients p1, p2, p3
P = np.linalg.solve(X, Y)

# Coefficients
p1, p2, p3 = P.flatten()

# Define the polynomial function
def f(x):
    return p1 * x**2 + p2 * x + p3

# Compute the predicted values for a smooth curve
x_smooth = np.linspace(min(x), max(x), 500)
y_smooth = f(x_smooth)

# Calculate f(350)
f_350 = f(350)

# Print the results
print("Coefficients:")
print(f"p1: {p1:.6f}")
print(f"p2: {p2:.6f}")
print(f"p3: {p3:.6f}")

print("\nPolynomial Equation:")
print(f"f(x) = {p1:.6f} * x^2 + {p2:.6f} * x + {p3:.6f}")

print(f"\nf(350) = {f_350:.6f}")

# Plot the data points and the interpolated curve
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_smooth, y_smooth, color='red', label='Interpolated polynomial')
plt.scatter(350, f_350, color='green', zorder=5)
plt.text(350, f_350, f"({350}, {f_350:.3f})", fontsize=9, ha='right')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Polynomial Interpolation')
plt.grid(True)
plt.show()

# Print the matrix form
print("\nMatrix Formulation:")
print(f"{X} * {np.array([['p1'], ['p2'], ['p3']])} = {Y}")

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')

# Prepare table data for display
table_data = [
    [f"{X[0, 0]:.0f}", f"{X[0, 1]:.0f}", f"{X[0, 2]:.0f}", "p1", f"{Y[0, 0]:.3f}"],
    [f"{X[1, 0]::.0f}", f"{X[1, 1]::.0f}", f"{X[1, 2]::.0f}", "p2", f"{Y[1, 0]:.3f}"],
    [f"{X[2, 0]::.0f}", f"{X[2, 1]::.0f}", f"{X[2, 2]::.0f}", "p3", f"{Y[2, 0]::.3f}"]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.show()
