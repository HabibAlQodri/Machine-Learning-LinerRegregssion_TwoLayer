# Analisis Putus Sekolah SD - Perbandingan Model Machine Learning

Repositori ini berisi implementasi dua model machine learning untuk menganalisis data putus sekolah SD: Regresi Linear dan Two-Layer Neural Network (Jaringan Saraf Tiruan Dua Lapis).

## Gambaran Umum

Proyek ini bertujuan untuk memprediksi dan menganalisis faktor-faktor yang mempengaruhi angka putus sekolah di tingkat Sekolah Dasar (SD) menggunakan dataset yang sama namun dengan pendekatan model yang berbeda.

## Persyaratan

- Python 3.x
- Pustaka yang diperlukan:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - tensorflow / keras (untuk model neural network)

## Dataset

Dataset yang digunakan adalah `dataset putus SD.csv` yang berisi informasi tentang faktor-faktor terkait dengan putus sekolah di tingkat SD.

```python
# Membaca dataset CSV
data = pd.read_csv('dataset putus SD.csv')

# Tampilkan 5 data teratas untuk pengecekan
print(data.head())
```

## Model 1: Regresi Linear

### Penjelasan

Regresi Linear adalah algoritma sederhana yang memodelkan hubungan linear antara variabel independen (fitur) dan variabel dependen (target). Model ini dapat direpresentasikan dengan persamaan: y = mx + b, di mana m adalah koefisien dan b adalah intercept.

### Implementasi

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Memisahkan fitur (X) dan target (Y)
X = data[['X']]
y = data['Y']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Linear Regression
model = LinearRegression()

# Melatih model dengan data training
model.fit(X_train, y_train)

# Menampilkan koefisien dan intercept
print('Koefisien:', model.coef_)
print('Intercept:', model.intercept_)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Visualisasi hasil regresi
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresi linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
```

### Keunggulan Regresi Linear

1. **Kesederhanaan**: Mudah diimplementasikan dan dipahami
2. **Efisiensi**: Pelatihan cepat dan ringan secara komputasi
3. **Interpretabilitas**: Parameter model (koefisien) memberikan informasi langsung tentang pengaruh setiap variabel
4. **Performa Baik**: Untuk hubungan yang memang linear, memberikan hasil yang akurat

### Keterbatasan Regresi Linear

1. **Asumsi Linearitas**: Hanya bisa menangkap hubungan linear antara variabel
2. **Kompleksitas Rendah**: Tidak bisa memodelkan interaksi kompleks atau pola non-linear
3. **Sensitif terhadap Outlier**: Performa dapat terpengaruh jika ada nilai ekstrem dalam data

## Model 2: Two-Layer Neural Network

### Penjelasan

Neural Network adalah model yang terinspirasi dari struktur otak manusia, terdiri dari neuron-neuron buatan yang terorganisir dalam layer. Two-Layer Neural Network memiliki dua layer tersembunyi antara input dan output, memungkinkan model untuk mempelajari pola non-linear yang kompleks.

### Implementasi

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Memisahkan fitur (X) dan target (Y)
X = data[['X']]
y = data['Y']

# Normalisasi data fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membuat model neural network dengan dua layer tersembunyi
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_scaled, y_scaled.ravel())

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Melatih model
y_pred_scaled = mlp.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Evaluasi model pada data uji
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error pada data uji: {mae:.4f}")

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Visualisasi hasil prediksi vs nilai sebenarnya
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Nilai Sebenarnya vs Prediksi')
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Nilai Prediksi')
plt.grid(True)
plt.show()
```

### Keunggulan Two-Layer Neural Network

1. **Fleksibilitas**: Mampu menangkap pola kompleks dan non-linear dalam data
2. **Kapasitas Tinggi**: Dapat beradaptasi dengan berbagai jenis hubungan data
3. **Performa**: Berpotensi memberikan akurasi prediksi yang lebih tinggi untuk data kompleks
4. **Generalisasi**: Dengan konfigurasi yang tepat, dapat menggeneralisasi dengan baik ke data baru

### Keterbatasan Two-Layer Neural Network

1. **Kompleksitas**: Lebih rumit untuk diimplementasikan dan dipahami
2. **Kebutuhan Data**: Umumnya membutuhkan lebih banyak data untuk pelatihan yang efektif
3. **Waktu Komputasi**: Membutuhkan waktu pelatihan yang lebih lama
4. **Interpretasi**: Sulit untuk menginterpretasikan parameter model (black box)
5. **Hyperparameter Tuning**: Memerlukan penyesuaian hyperparameter yang lebih kompleks

## Perbandingan Model

| Aspek | Regresi Linear | Two-Layer Neural Network |
|-------|---------------|-------------------------|
| Kompleksitas | Rendah | Tinggi |
| Waktu Pelatihan | Cepat | Lambat |
| Interpretabilitas | Tinggi | Rendah |
| Kemampuan Memodelkan Data Non-Linear | Rendah | Tinggi |
| Kebutuhan Data | Sedikit | Banyak |
| Kemampuan Generalisasi | Terbatas | Potensial Lebih Baik |
| Menangani Missing Values | Perlu Pre-processing | Perlu Pre-processing |
| Hyperparameter Tuning | Minimal | Ekstensif |

## Rekomendasi Penggunaan

### Gunakan Regresi Linear jika:
- Data menunjukkan hubungan yang cenderung linear
- Interpretabilitas model adalah prioritas
- Sumber daya komputasi terbatas
- Ukuran dataset relatif kecil
- Membutuhkan penjelasan tentang pengaruh setiap variabel

### Gunakan Two-Layer Neural Network jika:
- Data menunjukkan pola non-linear yang kompleks
- Akurasi prediksi adalah prioritas utama
- Memiliki dataset yang cukup besar
- Memiliki sumber daya komputasi yang memadai
- Interpretabilitas model bukan prioritas utama

## Cara Menjalankan Kode

1. Pastikan semua pustaka yang diperlukan sudah terpasang:
   ```
   pip install pandas numpy scikit-learn matplotlib tensorflow
   ```

2. Letakkan file 'dataset putus SD.csv' di direktori yang sama dengan skrip

3. Untuk menjalankan model Regresi Linear:
   ```
   python linear_regression_model.py
   ```

4. Untuk menjalankan model Two-Layer Neural Network:
   ```
   python neural_network_model.py
   ```

## Kesimpulan

Kedua model memiliki kekuatan dan keterbatasan masing-masing. Pemilihan model tergantung pada karakteristik data, kebutuhan interpretabilitas, dan sumber daya yang tersedia. Untuk hasil terbaik, disarankan untuk mencoba kedua pendekatan dan membandingkan performa mereka pada dataset putus sekolah SD.
