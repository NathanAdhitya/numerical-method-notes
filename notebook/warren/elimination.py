import numpy as np

# Define the coefficient matrix and the constant vector
A = np.array([[90000, 300, 1],
              [160000, 400, 1],
              [250000, 500, 1]], dtype=np.float64)

B = np.array([0.616, 0.525, 0.457], dtype=np.float64)

# Combine A and B into an augmented matrix
AB = np.column_stack((A, B.reshape(-1, 1)))

# Perform Gaussian elimination
n = len(B)
for i in range(n):
    # Partial pivoting for numerical stability (optional step)
    max_row = np.argmax(np.abs(AB[i:, i])) + i
    AB[[i, max_row]] = AB[[max_row, i]]
    
    # Eliminate the coefficients below the pivot
    for j in range(i + 1, n):
        factor = AB[j, i] / AB[i, i]
        AB[j, :] -= factor * AB[i, :]

# Perform back substitution to find the solution
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    x[i] = (AB[i, n] - np.dot(AB[i, i+1:n], x[i+1:])) / AB[i, i]

# Print the solution
print("Solution:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.4f}")