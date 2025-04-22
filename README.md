# Analisis Putus Sekolah SD - Model Regresi Linear

Repositori ini berisi kode untuk menganalisis data putus sekolah SD menggunakan model regresi linear sederhana.

## Gambaran Umum

Skrip ini melakukan operasi-operasi berikut:
1. Memuat data putus sekolah dari file CSV
2. Menyiapkan data dengan memisahkan fitur (X) dan variabel target (Y)
3. Membagi data menjadi set pelatihan dan pengujian
4. Melatih model regresi linear
5. Membuat prediksi dan mengevaluasi model
6. Memvisualisasikan hubungan antara variabel dengan plot scatter dan garis regresi

## Persyaratan

- Python 3.x
- Pustaka yang diperlukan:
  - pandas
  - scikit-learn
  - matplotlib

## Penjelasan Kode

### Pemuatan dan Persiapan Data
```python
# Membaca dataset CSV
data = pd.read_csv('dataset putus SD.csv')

# Tampilkan 5 data teratas untuk pengecekan
print(data.head())

# Memisahkan fitur (X) dan target (Y)
X = data[['X']]
y = data['Y']
```

### Pembagian Data
```python
# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Pelatihan Model
```python
# Membuat model Linear Regression
model = LinearRegression()

# Melatih model dengan data training
model.fit(X_train, y_train)

# Menampilkan koefisien dan intercept
print('Koefisien:', model.coef_)
print('Intercept:', model.intercept_)
```

### Prediksi dan Visualisasi
```python
# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menampilkan hasil prediksi
print('Hasil Prediksi:', y_pred)

# Visualisasi hasil regresi
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresi linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
```

## Cara Penggunaan

1. Pastikan semua pustaka yang diperlukan sudah terpasang
2. Letakkan file 'dataset putus SD.csv' di direktori yang sama dengan skrip
3. Jalankan skrip Python

## Catatan

Dalam kode ini, fitur X dan target Y diasumsikan sudah ada dalam dataset. Sesuaikan nama kolom sesuai dengan dataset aktual Anda.
