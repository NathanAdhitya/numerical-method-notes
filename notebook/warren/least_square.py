import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
X = np.array([10, 20, 30, 40, 50, 60, 70, 80])
Y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

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

# Buat tabel dengan pandas
data = {
    "X (Waktu)": X,
    "Y (Nilai)": Y,
    "Y_pred (Prediksi)": Y_pred,
    "X_i^2": X**2,
    "X_i * Y_i": X * Y,
    "a_0 + a_1x_i": Y_pred,
    "(y_i - ȳ)^2": (Y - Y_mean) ** 2,
    "(y_i - a_0 - a_1x_i)^2": (Y - Y_pred) ** 2
}

df = pd.DataFrame(data)

# Urutkan data berdasarkan nilai X
df_sorted = df.sort_values(by=["X (Waktu)"])

# Hitung jumlah (sigma) setiap kolom
sigma = df_sorted.sum()
sigma["X (Waktu)"] = "Σ"
sigma_df = pd.DataFrame(sigma).T

# Gabungkan data frame yang sudah diurutkan dengan sigma
df_sorted = pd.concat([df_sorted, sigma_df], ignore_index=True)

# Print tabel yang sudah diurutkan
print("\nTabel Data yang Diurutkan:")
print(df_sorted)

# Tampilkan tabel menggunakan matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_sorted.values, colLabels=df_sorted.columns, cellLoc='center', loc='center')

# Ubah ukuran font pada tabel agar sesuai
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Display the table
plt.show()
