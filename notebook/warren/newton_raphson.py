import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, diff

# Definisi simbol-simbol yang digunakan
x1, x2 = symbols('x1 x2')

# Definisi fungsi nonlinier
# f1 = x1**2 + x1*x2 - 10
# f2 = x2 + 3*x1*x2**2 - 57
f1 = x2 + 3*x1**2 + 5*x1 -2
f2 = 5*x1*x2 - 3 - x1**2

# Gabungkan kedua fungsi menjadi satu fungsi yang dapat dipanggil
def func(x):
    return [f1.subs({x1: x[0], x2: x[1]}), f2.subs({x1: x[0], x2: x[1]})]

# Hitung matriks Jacobian dalam bentuk simbolik
J_symbolic = np.array([[diff(f1, x1), diff(f1, x2)], [diff(f2, x1), diff(f2, x2)]])

# Evaluasi matriks Jacobian ke dalam bentuk numerik
J_numeric = np.array([[J_symbolic[0, 0].evalf(subs={x1: -2, x2: 0}), J_symbolic[0, 1].evalf(subs={x1: -2, x2: 0})],
                      [J_symbolic[1, 0].evalf(subs={x1: -2, x2: 0}), J_symbolic[1, 1].evalf(subs={x1: -2, x2: 0})]])

# Konversi matriks Jacobian ke dalam tipe data float64
J = np.array(J_numeric, dtype=np.float64)

# Hitung determinan matriks Jacobian
det_J = np.linalg.det(J)

print("Matriks Jacobian:")
print(J)
print("Determinan Matriks Jacobian:", det_J)

# Nilai tebakan awal
x0 = np.array([1.0, 1.0])  # Tebakan awal baru

# Solusi sistem nonlinier dengan metode Newton-Raphson
sol = fsolve(func, x0)

# Tampilkan solusi x1 dan x2 dibagi dengan determinan
sol_divided_by_det = sol / det_J
print("\nSolusi x yang Dibagi dengan Determinan Matriks Jacobian:")
print(sol_divided_by_det)

# Hitung turunan parsial pada solusi
partial_derivatives = [[diff(f1, x1).subs({x1: sol[0], x2: sol[1]}), diff(f1, x2).subs({x1: sol[0], x2: sol[1]})],
                       [diff(f2, x1).subs({x1: sol[0], x2: sol[1]}), diff(f2, x2).subs({x1: sol[0], x2: sol[1]})]]

print("\nTurunan Parsial pada Solusi:")
print(partial_derivatives)