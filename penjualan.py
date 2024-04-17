import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Baca dataset
data = pd.read_csv('DataPenjualan.csv')

# Pisahkan fitur (X) dan target (y)
X = data[['jumlah_penjualan', 'harga']]
y = data['merk_produk']

# Label Encoding untuk kolom target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Inisialisasi model regresi linear
model = LinearRegression()

# Latih model
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('Coefficient of Determination (R^2):', r2)

# Ubah nilai harga untuk nilai sebenarnya dan nilai prediksi
harga_sebenarnya = X_test['harga']
harga_prediksi = y_pred

# Visualisasi hasil prediksi
plt.figure(figsize=(8, 6))
plt.scatter(harga_sebenarnya, harga_prediksi, label='Predicted')
plt.plot([min(harga_sebenarnya), max(harga_sebenarnya)], [min(harga_sebenarnya), max(harga_sebenarnya)], color='red', linestyle='-', linewidth=2, label='45° Line')
plt.xlabel('Harga Sebenarnya')
plt.ylabel('Harga Diprediksi')
plt.title('Harga Sebenarnya vs Harga Diprediksi')
plt.legend()
plt.grid(True)
plt.show()
