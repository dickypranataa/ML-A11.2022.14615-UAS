# Prediksi Harga Bahan Pokok Menggunakan XGBoost dan Linear Regression

## Identitas Lengkap
- **Nama:** [Nama Lengkap Anda]
- **NIM:** [NIM Anda]
- **Mata Kuliah:** [Nama Mata Kuliah]
- **Dosen:** [Nama Dosen]

---

## 1. Ringkasan dan Permasalahan Project

### Ringkasan
Proyek ini bertujuan untuk memprediksi harga bahan pokok menggunakan algoritma XGBoost dan Linear Regression. Data yang digunakan adalah harga bulanan dari berbagai bahan pokok. Model dibangun untuk memprediksi harga di masa depan berdasarkan data historis.

### Permasalahan
Harga bahan pokok seringkali mengalami fluktuasi yang signifikan karena berbagai faktor ekonomi dan sosial. Memiliki model yang dapat memprediksi harga di masa depan akan sangat berguna untuk perencanaan dan pengambilan keputusan.

### Tujuan
1. Membangun model prediksi harga bahan pokok menggunakan XGBoost dan Linear Regression.
2. Membandingkan kinerja model yang dibangun.
3. Mengidentifikasi fitur-fitur penting yang mempengaruhi harga bahan pokok.

### Model / Alur Penyelesaian

![Alur Penyelesaian](path_to_your_flowchart_image.png)

---

## 2. Penjelasan Dataset, EDA dan Proses Features Dataset

### Penjelasan Dataset
Dataset yang digunakan adalah data harga bulanan dari berbagai bahan pokok. Dataset ini terdiri dari kolom-kolom sebagai berikut:
- `bahan_pokok`: Nama bahan pokok
- `januari` hingga `desember`: Harga bahan pokok setiap bulan

### Eksplorasi Data Awal (EDA)
Berikut adalah beberapa visualisasi dan statistik dasar dari dataset:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baca dataset
data = pd.read_csv('DataPenjualan.csv')

# Tampilkan 5 baris pertama
data.head()

# Statistik deskriptif
data.describe()

# Visualisasi distribusi harga
plt.figure(figsize=(12, 6))
sns.boxplot(x='bahan_pokok', y='januari', data=data)
plt.xticks(rotation=90)
plt.title('Distribusi Harga Bahan Pokok di Bulan Januari')
plt.show()
