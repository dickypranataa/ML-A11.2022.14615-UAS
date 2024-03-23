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

# Visualisasi hasil prediksi
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', linewidth=2)  # Garis 45 derajat
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Nilai Diprediksi')
plt.title('Nilai Sebenarnya vs Nilai Diprediksi')
plt.show()

