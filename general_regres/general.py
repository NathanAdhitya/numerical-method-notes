import numpy as np


# copas dari excel
N = np.array([[6,15,55],[15,55,225],[55,225,979]])
r = np.array([152.6,585.6,2488.8])

# cara cepat
np.linalg.solve(N, r)


# pakai cara
# 1. invers N
# 2. kali dot N invers dengan r
N_inv = np.linalg.inv(N)
print("N invers")
print(N_inv)
print("hasil akhr")
x = np.dot(N_inv, r)
# a0,a1,a2
print(x)

# fungsi akhir regresi
# f(x) = a0 + a1*x + a2*x^2