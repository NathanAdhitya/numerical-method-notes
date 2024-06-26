import numpy as np

def jacobi(A, b, x0, max_iterations=100, tolerance=1e-6):
    n = len(A)
    x = x0.copy()
    iterations = 0

    while iterations < max_iterations:
        x_new = np.zeros_like(x)

        for i in range(n):
            sum_val = 0
            for j in range(n):
                if j != i:
                    sum_val += A[i][j] * x[j]
            x_new[i] = round((b[i] - sum_val) / A[i][i], 3)

        print(f"Iteration {iterations + 1}: {x_new}")

        if np.linalg.norm(x_new - x) < tolerance:
            break

        x = x_new
        iterations += 1

    return x

# Example usage
A = np.array([[5, -1, 2], [3, 8, -2], [1, 1, 4]])
b = np.array([12, -25, 6])
x0 = np.array([0.0, 0.0, 0.0])

solution = jacobi(A, b, x0)
print("Solution:", solution)
