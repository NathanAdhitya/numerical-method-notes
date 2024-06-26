import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])

# Number of data points
n = len(x)

# Formulate the design matrix for a quadratic fit
X = np.vstack((np.ones(n), x, x**2)).T
Y = y.reshape(-1, 1)

# Compute the transpose of X
X_transpose = X.T

# Compute the normal equation components
X_transpose_X = np.dot(X_transpose, X)
X_transpose_Y = np.dot(X_transpose, Y)

# Solve for the coefficients a0, a1, a2
A = np.linalg.solve(X_transpose_X, X_transpose_Y)

# Coefficients
a0, a1, a2 = A.flatten()

# Compute the predicted values
y_pred = a0 + a1 * x + a2 * x**2

# Compute the residuals
residuals = y - y_pred

# Compute the mean of y
y_mean = np.mean(y)

# Compute the various sums needed
S_y_ybar_sq = np.sum((y - y_mean) ** 2)
S_y_fit_sq = np.sum((residuals) ** 2)

# Compute standard error of the estimate
s_y_x = np.sqrt(S_y_fit_sq / (n - 3))

# Compute the coefficient of determination
r_squared = 1 - (S_y_fit_sq / S_y_ybar_sq)

# Compute the correlation coefficient
r = np.sqrt(r_squared)

# Prepare the table data
data = {
    'xi': x,
    'yi': y,
    '(yi - y_bar)^2': (y - y_mean) ** 2,
    '(yi - a0 - a1*xi - a2*xi^2)^2': residuals ** 2
}

df = pd.DataFrame(data)
df.loc['Σ'] = df.sum(numeric_only=True)
df.at['Σ', 'xi'] = 'Σ'
df.at['Σ', 'yi'] = 'Σ'

# Print the matrices
print("Design matrix (X):")
print(X)
print("\nObserved values vector (Y):")
print(Y)
print("\nX^T * X:")
print(X_transpose_X)
print("\nX^T * Y:")
print(X_transpose_Y)
print("\nCoefficients vector (A):")
print(A)

# Print the results
print("\nCoefficients:")
print(f"a0: {a0:.4f}")
print(f"a1: {a1:.4f}")
print(f"a2: {a2:.4f}")
print("\nTable:")
print(df)
print(f"\nStandard error of the estimate (sy/x): {s_y_x:.4f}")
print(f"Coefficient of determination (r^2): {r_squared:.4f}")
print(f"Correlation coefficient (r): {r:.4f}")

# Plot the data points and the fitted curve
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Quadratic fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Display the matrix as in the provided image
print("\nMatrix Formulation:")
print(f"{X_transpose_X} * {np.array([['a0'], ['a1'], ['a2']])} = {X_transpose_Y}")

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')

# Prepare table data for display
table_data = [
    [f"{X_transpose_X[0, 0]:.0f}", f"{X_transpose_X[0, 1]:.0f}", f"{X_transpose_X[0, 2]:.0f}", "a0", f"{X_transpose_Y[0, 0]:.1f}"],
    [f"{X_transpose_X[1, 0]:.0f}", f"{X_transpose_X[1, 1]:.0f}", f"{X_transpose_X[1, 2]:.0f}", "a1", f"{X_transpose_Y[1, 0]:.1f}"],
    [f"{X_transpose_X[2, 0]:.0f}", f"{X_transpose_X[2, 1]:.0f}", f"{X_transpose_X[2, 2]:.0f}", "a2", f"{X_transpose_Y[2, 0]:.1f}"]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.show()
