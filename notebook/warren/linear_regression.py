import numpy as np

import matplotlib.pyplot as plt

# Data
X = np.array([4.3, 6.4, 4.5, 1.7, 4.5, 5.6, 1.2, 2.3, 6.1, 3.1, 4.8, 3.5])
Y = np.array([85, 54, 77, 90, 64, 65, 90, 83, 68, 73, 68, 83])

# Hitung rata-rata
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Hitung slope (beta_1) dan intercept (beta_0)
beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
beta_0 = Y_mean - beta_1 * X_mean

# Persamaan garis regresi
def predict(x):
    return beta_0 + beta_1 * x

# Hitung prediksi
Y_pred = predict(X)
# Hitung sy (total standard deviation)
s_y = np.sqrt(np.sum((Y - Y_mean) ** 2) / (len(Y) - 1))

# Hitung sy/x (standard error of the estimate)
s_y_x = np.sqrt(np.sum((Y - Y_pred) ** 2) / (len(Y) - 2))

# Hitung r^2 (coefficient of determination)
r_squared = np.sum((Y_pred - Y_mean) ** 2) / np.sum((Y - Y_mean) ** 2)

# Hitung r (correlation coefficient)
r = np.sqrt(r_squared)

# Plot data dan garis regresi
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, Y_pred, color="red", label="Garis Regresi")
plt.xlabel("Waktu (Jam/Hari)")
plt.ylabel("Nilai")
plt.legend()

# Tambahkan teks pada setiap titik data
for i in range(len(X)):
    plt.text(X[i], Y[i], f"({X[i]}, {Y[i]})")

# Print statements
print(f"Persamaan garis regresi: Y = {beta_0:.2f} + {beta_1:.2f}X")
print(f"Total standard deviation (sy): {s_y:.2f}")
print(f"Standard error of the estimate (sy/x): {s_y_x:.2f}")
print(f"Coefficient of determination (r^2): {r_squared:.2f}")
print(f"Correlation coefficient (r): {r:.2f}")

# Display the plot
plt.show()

# Plot data dan garis regresi
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, Y_pred, color="red", label="Garis Regresi")
plt.xlabel("Waktu (Jam/Hari)")
plt.ylabel("Nilai")
plt.legend()

# Tambahkan teks pada setiap titik data
for i in range(len(X)):
    plt.text(X[i], Y[i], f"({X[i]}, {Y[i]})")

plt.show()


