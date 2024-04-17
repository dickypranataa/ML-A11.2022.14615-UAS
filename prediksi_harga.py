import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Baca dataset
data = pd.read_csv('DataPenjualan.csv')

# Inisialisasi dictionary untuk menyimpan model untuk setiap merk_produk
models = {}

# Lakukan training model untuk setiap merk_produk
for merk_produk in data['merk_produk'].unique():
    # Filter data untuk merk_produk tertentu
    data_merk = data[data['merk_produk'] == merk_produk]
    
    # Pisahkan fitur (X) dan target (y)
    X = data_merk[['jumlah_penjualan', 'harga']]
    y = data_merk['harga']  # Harga akan diprediksi
    
    # Latih model regresi linear
    model = LinearRegression()
    model.fit(X, y)
    
    # Simpan model dalam dictionary
    models[merk_produk] = model

# Visualisasi prediksi harga untuk setiap merk_produk
plt.figure(figsize=(10, 6))
for merk_produk, model in models.items():
    # Filter data untuk merk_produk tertentu
    data_merk = data[data['merk_produk'] == merk_produk]
    
    # Pisahkan fitur (X) untuk visualisasi
    X_vis = data_merk[['jumlah_penjualan', 'harga']]
    
    # Prediksi harga menggunakan model yang sesuai
    prediksi_harga = model.predict(X_vis)
    
    # Plot data aktual
    plt.scatter(data_merk['jumlah_penjualan'], data_merk['harga'], label=merk_produk)
    
    # Plot prediksi harga
    plt.plot(data_merk['jumlah_penjualan'], prediksi_harga, linestyle='--')
    
plt.xlabel('Jumlah Penjualan')
plt.ylabel('Harga')
plt.title('Prediksi Harga untuk Setiap Merk Produk')
plt.legend()
plt.grid(True)
plt.show()
