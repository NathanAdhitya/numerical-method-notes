import numpy as np

# Masukkan matriks pertama
matrix1 = np.array(
    [
        [1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 4, 9, 16, 25],
    ]
)

# Masukkan matriks kedua
matrix2 = np.array([[1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16], [1, 5, 25]])

# Inisialisasi matriks hasil dengan nol
result_matrix = np.zeros((matrix1.shape[0], matrix2.shape[1]))

# Tampilkan langkah-langkah perkalian matriks
print("Langkah-langkah Perkalian Matriks:")
for i in range(matrix1.shape[0]):
    for j in range(matrix2.shape[1]):
        for k in range(matrix1.shape[1]):
            result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
        print(f"Langkah {i+1}-{j+1}:")
        print(matrix1[i], "x", matrix2[:, j], "=", result_matrix[i][j])
        print("Hasil sementara:")
        print(result_matrix)

print("\nHasil Perkalian Matriks Akhir:")
print(result_matrix)
