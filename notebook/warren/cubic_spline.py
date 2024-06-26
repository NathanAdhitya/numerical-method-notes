import numpy as np
import matplotlib.pyplot as plt

# Data dari gambar
x = np.array([-2.5, -1.5, 0.5, 2.5])  # Nodes
y = np.array([4.5, -1.5, 2.5, 0.5])  # Function values at nodes

# Banyak interval
n = len(x) - 1

# Matriks A dan vektor b untuk sistem persamaan
A = np.zeros((4 * n, 4 * n))
b = np.zeros(4 * n)

# Mengisi matriks A dan vektor b berdasarkan kondisi kontinuitas
row = 0

# Kondisi kontinuitas (fungsi harus melewati semua titik)
for i in range(n):
    A[row, 4 * i:4 * i + 4] = [x[i]**3, x[i]**2, x[i], 1]
    b[row] = y[i]
    row += 1
    A[row, 4 * i:4 * i + 4] = [x[i + 1]**3, x[i + 1]**2, x[i + 1], 1]
    b[row] = y[i + 1]
    row += 1

# Kondisi kontinuitas turunan pertama di node interior
for i in range(1, n):
    A[row, 4 * (i - 1):4 * (i - 1) + 4] = [3 * x[i]**2, 2 * x[i], 1, 0]
    A[row, 4 * i:4 * i + 4] = [-3 * x[i]**2, -2 * x[i], -1, 0]
    row += 1

# Kondisi kontinuitas turunan kedua di node interior
for i in range(1, n):
    A[row, 4 * (i - 1):4 * (i - 1) + 4] = [6 * x[i], 2, 0, 0]
    A[row, 4 * i:4 * i + 4] = [-6 * x[i], -2, 0, 0]
    row += 1

# Turunan kedua di endpoint adalah nol
A[row, 0:4] = [6 * x[0], 2, 0, 0]
row += 1
A[row, -4:] = [6 * x[-1], 2, 0, 0]

# Menyelesaikan sistem persamaan
coeffs = np.linalg.solve(A, b)

# Menampilkan koefisien
for i in range(n):
    print(f"Spline interval {i}:")
    print(f"   a{i} = {coeffs[4 * i]:.4f}")
    print(f"   b{i} = {coeffs[4 * i + 1]:.4f}")
    print(f"   c{i} = {coeffs[4 * i + 2]:.4f}")
    print(f"   d{i} = {coeffs[4 * i + 3]:.4f}")

# Fungsi untuk mengevaluasi spline kubik
def cubic_spline_eval(x_val, coeffs, x_nodes):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            a, b, c, d = coeffs[4 * i:4 * i + 4]
            return a * x_val**3 + b * x_val**2 + c * x_val + d
    return None

# Evaluasi spline pada rentang nilai x
x_range = np.linspace(min(x), max(x), 200)
y_range = [cubic_spline_eval(xi, coeffs, x) for xi in x_range]

# Plot hasil
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Titik asli')
plt.plot(x_range, y_range, '-', label='Spline kubik')
plt.legend()
plt.title('Interpolasi Spline Kubik')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
