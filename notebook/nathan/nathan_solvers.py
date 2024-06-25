from typing import Callable
import numpy as np
import pandas as pd


def newton_raphson(f: Callable, x0: np.ndarray, es=0.00001, max_iter=50):
    """
    Newton-Raphson method to solve nonlinear equations
        f = function, should return pair of function and derivative results
        x0 = initial guesses
        es = stop criterion (default 0.00001)
        max_iter = maximum number of iterations (default 50)
    output:
    x = solution
    """
    x = np.copy(x0)
    it = 0
    while True:
        f_x, jf_x = f(x)

        print(f"Iteration {it+1}:")

        for i in range(len(f_x)):
            ni = i + 1
            for j in range(len(jf_x[i])):
                ji = j + 1
                print(f"df{ni}/x{ji} = {jf_x[i][j]}")

        # Print jacobian
        jacobian = np.linalg.det(jf_x)
        print(f"J = {jacobian}")

        # multiplying by the inverse of the jacobian
        dx = np.matmul(np.linalg.inv(jf_x), f_x)
        x = x - dx
        it += 1
        ea = 100 * np.max(np.abs(dx / x))

        print(f"x = {x}, ea = {ea}")
        print()

        if ea < es or it >= max_iter:
            print(f"Took {it} iterations to converge")
            break
    return x
