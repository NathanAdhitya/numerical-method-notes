import numpy as np

def gauss_seidel(A, b, x0, tolerance=1e-10, max_iterations=1000):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.
    :param A: Coefficient matrix
    :param b: Constant terms vector
    :param x0: Initial guess for the solution
    :param tolerance: Tolerance for the convergence criterion
    :param max_iterations: Maximum number of iterations
    :return: Solution vector and a list of (iteration, x) for each iteration
    """
    x = x0.copy()
    iterations = []
    n = A.shape[0]
    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
            print(f"Iteration: {k}, x[{i}]: {x[i]}")  # Print each computational step
        iterations.append((k, x.copy()))
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break
    return x, iterations

# Example usage
if __name__ == "__main__":
    A = np.array([[5, -1, 2],
                  [3, 8, -2],
                  [1, 1, 4],], dtype=float)
    b = np.array([12, -25 ,6], dtype=float)
    x0 = np.zeros(A.shape[0])

    solution, iterations = gauss_seidel(A, b, x0)

    # Printing the iterations
    print(f"{'Iteration':>10} {'x':>30}")
    for k, x in iterations:
        print(f"{k:10d} {np.array2string(x, precision=4)}")

    # Printing the final solution
    print(f"Solution: {np.array2string(solution, precision=4)}")
