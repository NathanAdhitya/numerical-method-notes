import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Nicsin's Code.


def gauss_seidel_fixed(A, b, tolerance=1e-10, max_iterations=3):
    # Initialize variables
    x1, x2, x3 = 0, 0, 0  # Initial guesses
    n = len(b)

    print("Initial guess: x1 = 0, x2 = 0, x3 = 0")

    for iteration in range(max_iterations):
        print("======================================================")
        x1_old, x2_old, x3_old = x1, x2, x3  # Save old values

        # Calculate x1
        x1 = (b[0] - A[0][1] * x2_old - A[0][2] * x3_old) / A[0][0]
        print(f"Iteration {iteration + 1}:")
        print(f"x1 = (b[0] - A[0][1] * x2_old - A[0][2] * x3_old) / A[0][0]")
        print(
            f"x1 = ({b[0]} - {A[0][1]} * {x2_old} - {A[0][2]} * {x3_old}) / {A[0][0]}"
        )
        print(f"x1 = {x1}\n")

        # Calculate x2
        x2 = (b[1] - A[1][0] * x1 - A[1][2] * x3_old) / A[1][1]
        print(f"x2 = (b[1] - A[1][0] * x1 - A[1][2] * x3_old) / A[1][1]")
        print(f"x2 = ({b[1]} - {A[1][0]} * {x1} - {A[1][2]} * {x3_old}) / {A[1][1]}")
        print(f"x2 = {x2}\n")

        # Calculate x3
        x3 = (b[2] - A[2][0] * x1 - A[2][1] * x2) / A[2][2]
        print(f"x3 = (b[2] - A[2][0] * x1 - A[2][1] * x2) / A[2][2]")
        print(f"x3 = ({b[2]} - {A[2][0]} * {x1} - {A[2][1]} * {x2}) / {A[2][2]}")
        print(f"x3 = {x3}\n")

        # Check for convergence
        if np.allclose([x1, x2, x3], [x1_old, x2_old, x3_old], atol=tolerance):
            print(f"Converged after {iteration + 1} iterations\n")
            return np.array([x1, x2, x3])

    print("Maximum iterations reached without convergence")
    return np.array([x1, x2, x3])


def jacobi_fixed(A, b, tolerance=1e-10, max_iterations=3):
    # Initialize variables
    x1, x2, x3 = 0, 0, 0  # Initial guesses
    n = len(b)

    print("Initial guess: x1 = 0, x2 = 0, x3 = 0")

    for iteration in range(max_iterations):
        print("======================================================")

        x1_old, x2_old, x3_old = x1, x2, x3  # Save old values

        print(f"Iteration {iteration + 1}:")

        # Calculate x1
        x1 = (b[0] - A[0][1] * x2_old - A[0][2] * x3_old) / A[0][0]
        print(f"x1 = (b[0] - A[0][1] * x2_old - A[0][2] * x3_old) / A[0][0]")
        print(
            f"x1 = ({b[0]} - {A[0][1]} * {x2_old} - {A[0][2]} * {x3_old}) / {A[0][0]}"
        )
        print(f"x1 = {x1}\n")

        # Calculate x2
        x2 = (b[1] - A[1][0] * x1_old - A[1][2] * x3_old) / A[1][1]
        print(f"x2 = (b[1] - A[1][0] * x1_old - A[1][2] * x3_old) / A[1][1]")
        print(
            f"x2 = ({b[1]} - {A[1][0]} * {x1_old} - {A[1][2]} * {x3_old}) / {A[1][1]}"
        )
        print(f"x2 = {x2}\n")

        # Calculate x3
        x3 = (b[2] - A[2][0] * x1_old - A[2][1] * x2_old) / A[2][2]
        print(f"x3 = (b[2] - A[2][0] * x1_old - A[2][1] * x2_old) / A[2][2]")
        print(
            f"x3 = ({b[2]} - {A[2][0]} * {x1_old} - {A[2][1]} * {x2_old}) / {A[2][2]}"
        )
        print(f"x3 = {x3}\n")

        # Check for convergence
        if np.allclose([x1, x2, x3], [x1_old, x2_old, x3_old], atol=tolerance):
            print(f"Converged after {iteration + 1} iterations\n")
            return np.array([x1, x2, x3])

    print("Maximum iterations reached without convergence")
    return np.array([x1, x2, x3])


def linear_regression(x, y):
    n = len(x)

    # Calculations
    xi_sum = np.sum(x)
    yi_sum = np.sum(y)
    xi_squared_sum = np.sum(x**2)
    xiyi_sum = np.sum(x * y)

    x_mean = xi_sum / n
    y_mean = yi_sum / n

    a1 = (n * xiyi_sum - xi_sum * yi_sum) / (n * xi_squared_sum - xi_sum**2)
    a0 = y_mean - a1 * x_mean

    y_pred = a0 + a1 * x

    St = np.sum((y - y_mean) ** 2)
    Sr = np.sum((y - y_pred) ** 2)
    sy = np.sqrt(St / (n - 1))
    sy_x = np.sqrt(Sr / (n - 2))
    r2 = 1 - (Sr / St)
    r = np.sqrt(r2)

    # Creating tables
    table1 = pd.DataFrame({"xi": x, "yi": y, "xi^2": x**2, "xi.yi": x * y})
    sums = table1.sum()
    averages = table1.mean()
    table1.loc["Sum"] = sums
    table1.loc["Average"] = averages

    table2 = pd.DataFrame({"xi": x, "yi": y, "yl (predicted)": y_pred})

    table3 = pd.DataFrame(
        {
            "xi": x,
            "yi": y,
            "a0+a1*xi": y_pred,
            "(yi-y_ave)^2": (y - y_mean) ** 2,
            "(yi-a0-a1*xi)^2": (y - y_pred) ** 2,
        }
    )
    sums = table3.sum()
    averages = table3.mean()
    table3.loc["Sum"] = sums
    table3.loc["Average"] = averages

    # Printing the tables
    print("Table 1:")
    print(table1, "\n")

    print("Table 2:")
    print(table2, "\n")

    print("Table 3:")
    print(table3, "\n")

    # Summary Statistics
    print("\nSummary Statistics:")
    print(f"sy (total standard deviation): {sy:.4f}")
    print(f"sy/x (standard error of the estimate): {sy_x:.4f}")
    print(f"r^2 (coefficient of determination): {r2:.4f}")
    print(f"r (correlation coefficient): {r:.4f}")
    print(f"Regression line: y = {a0:.4f} + {a1:.4f}x")

    # Plotting the results
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, y_pred, color="red", label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression")
    plt.show()
