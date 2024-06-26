import numpy as np

# Data titik (xi, fi)
data = np.array([
    [3.0, 2.5],
    [4.5, 1.0],
    [7.0, 2.5],
    [9.0, 0.5]
])

# Hitung interval h dan selisih f
h = np.diff(data[:, 0])
f = data[:, 1]

# Matriks koefisien
A = np.array([
    [h[0], 0, 0, 0, 0],
    [0, h[1], h[1]**2, 0, 0],
    [0, 0, 0, h[2], h[2]**2],
    [1, -1, 0, 0, 0],
    [0, 1, 2*h[1], -1, 0]
])

# Vektor hasil
b = np.array([
    f[1] - f[0],
    f[2] - f[1],
    f[3] - f[2],
    0,
    0
])

# Menyelesaikan sistem persamaan linear
coefficients = np.linalg.solve(A, b)
b1, b2, c2, b3, c3 = coefficients

# Tampilkan matriks dan vektor hasil
print("Matriks koefisien (A):")
print(A)
print("\nVektor hasil (b):")
print(b)
print("\nHasil penyelesaian:")
print(f"b1 = {b1}")
print(f"b2 = {b2}")
print(f"c2 = {c2}")
print(f"b3 = {b3}")
print(f"c3 = {c3}")

# Definisikan fungsi spline
def s1(x):
    return f[0] + b1 * (x - data[0, 0])

def s2(x):
    return f[1] + b2 * (x - data[1, 0]) + c2 * (x - data[1, 0])**2

def s3(x):
    return f[2] + b3 * (x - data[2, 0]) + c3 * (x - data[2, 0])**2

x_val = 5

# Tentukan interval dan hitung nilai spline
if data[0, 0] <= x_val < data[1, 0]:
    result = s1(x_val)
elif data[1, 0] <= x_val < data[2, 0]:
    result = s2(x_val)
elif data[2, 0] <= x_val <= data[3, 0]:
    result = s3(x_val)
else:
    result = None
    print("x is out of the interpolation range")

print(f"\nNilai spline kuadratik di x = {x_val} adalah {result}")
