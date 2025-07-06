# Proyek-Predictive-Analytics
# Laporan Proyek Machine Learning - Maulana Ridhwan Riziq

## Domain Proyek

Pertumbuhan populasi perkotaan dan meningkatnya jumlah kendaraan setiap tahunnya menyebabkan kemacetan lalu lintas menjadi salah satu permasalahan utama di kota-kota besar. Lalu lintas yang padat dapat berdampak signifikan terhadap produktivitas masyarakat, polusi udara, serta konsumsi bahan bakar. Oleh karena itu, dibutuhkan sistem prediksi volume lalu lintas yang akurat untuk membantu pengambilan keputusan oleh otoritas transportasi dalam melakukan pengelolaan lalu lintas.

Prediksi volume lalu lintas secara akurat dapat digunakan untuk:

* Mengoptimalkan pengaturan lampu lalu lintas.
* Menentukan waktu terbaik untuk melakukan perawatan jalan.
* Mengatur pola kerja berbasis lalu lintas.
* Memberikan informasi proaktif kepada pengguna jalan.

Dengan kemajuan teknologi dan ketersediaan data lalu lintas secara real-time, pendekatan machine learning dan deep learning menjadi metode yang menjanjikan dalam menyelesaikan permasalahan prediksi lalu lintas. LSTM (Long Short-Term Memory), sebagai salah satu jenis Recurrent Neural Network (RNN), memiliki kemampuan dalam memproses data urutan (sequence) dan telah terbukti efektif dalam menyelesaikan berbagai kasus time series forecasting \[1]\[2].

## Business Understanding

### Problem Statements

* Bagaimana memodelkan dan memprediksi volume lalu lintas secara akurat berdasarkan data historis yang tersedia?
* Bagaimana pengaruh fitur waktu dan cuaca terhadap volume lalu lintas?

### Goals

* Membuat model prediksi volume lalu lintas berbasis data historis menggunakan LSTM.
* Menganalisis pengaruh variabel waktu dan cuaca terhadap tingkat kepadatan lalu lintas.

### Solution Statements

* Menggunakan model LSTM untuk memprediksi volume lalu lintas berdasarkan fitur waktu, kondisi cuaca, dan status libur.
* Melakukan hyperparameter tuning untuk meningkatkan akurasi prediksi.

## Data Understanding

Dataset yang digunakan adalah **Metro Interstate Traffic Volume** yang tersedia di Kaggle \[3]. Dataset ini berisi data lalu lintas dari sensor jalan raya Interstate 94 di Minneapolis, Minnesota, dikumpulkan dari tahun 2012 hingga 2018.

### Ringkasan Dataset:

* **Jumlah data:** 48.204 baris dan 9 kolom.
* **Variabel Target:** `traffic_volume` (jumlah kendaraan per jam).
* **Fitur Waktu:** `date_time`.
* **Fitur Cuaca:** `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`.
* **Fitur Hari Libur:** `holiday`.

### Pemeriksaan Data:

* **Missing Value:**

  * Kolom holiday hanya memiliki 61 nilai non-null dari total 48.204 baris, yang berarti sebagian besar nilainya adalah missing (NaN),  
  * Kolom lain tidak memiliki missing values.
* **Outliers:**

  * Terdapat nilai volume lalu lintas ekstrem rendah (<100) yang sangat jarang, kemungkinan terjadi saat malam atau dini hari.
* **Duplikasi:** Tidak ditemukan data duplikat berdasarkan `date_time`.

Visualisasi awal menunjukkan adanya pola harian dan mingguan pada `traffic_volume`, serta hubungan non-linear dengan fitur cuaca seperti suhu dan kondisi cuaca.

## Data Preparation

Langkah-langkah yang dilakukan pada tahap persiapan data:

1. **Konversi kolom waktu**

   * Mengubah kolom `date_time` menjadi tipe `datetime`.
   * Menjadikan `date_time` sebagai indeks.
   * Mengurutkan data berdasarkan indeks waktu.

2. **Konversi suhu**

   * Mengonversi nilai pada kolom `temp` dari Kelvin ke Celcius menggunakan rumus: `df['temp'] = df['temp'] - 273.15`

3. **Pembersihan Data**

   * Mengecek dan menangani missing value.
   * Kolom `holiday` yang mayoritas nilainya missing, ditangani dengan one-hot encoding sehingga missing value tidak menyebabkan error dalam pemodelan.

4. **Feature Engineering**

   * Ekstraksi fitur waktu: `hour`, `day`, `dayofweek`, `is_weekend` dari kolom indeks `date_time`.
   * Encoding fitur kategorikal `weather_main` dan `holiday` menggunakan one-hot encoding.

5. **Pembagian Data (Train-Test Split)**

   * Data dibagi berdasarkan urutan waktu: 80% data pertama sebagai data latih (`df_train`) dan 20% sisanya sebagai data uji (`df_test`).

6. **Normalisasi Data**

   * Menggunakan Min-Max Scaler pada semua fitur numerik (kecuali target) untuk mempercepat dan menstabilkan proses pelatihan model.

7. **Windowing (Membuat jendela input-output)**

   * Format data diubah menjadi bentuk sequence dengan window sepanjang 24 jam sebagai input dan 1 jam ke depan sebagai target.

## Modeling

Model yang digunakan dalam proyek ini adalah Long Short-Term Memory (LSTM), yang efektif dalam menangani data time series dan mampu mengingat pola jangka panjang.

Cara Kerja LSTM

LSTM adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient dan kesulitan dalam mengingat informasi jangka panjang. LSTM memiliki struktur khusus yang disebut cell dan didalamnya terdapat empat komponen utama:

1. Forget Gate: Memutuskan informasi mana dari cell state sebelumnya yang perlu dibuang. Menggunakan sigmoid layer untuk menghasilkan nilai antara 0 (buang) dan 1 (simpan).
2. Input Gate: Memutuskan informasi baru apa yang akan disimpan ke dalam cell state. Terdiri dari sigmoid layer dan tanh layer untuk memilih dan menambahkan nilai baru.
3. Cell State: Merupakan memori utama dari jaringan LSTM. Informasi dari forget gate dan input gate digunakan untuk memperbarui nilai cell state.
4. Output Gate: Menentukan informasi apa yang akan dikeluarkan dari sel saat ini sebagai output dan juga diteruskan ke langkah waktu berikutnya.

Dengan adanya cell state dan mekanisme gate tersebut, LSTM mampu menyimpan informasi penting dan melupakan informasi yang tidak relevan secara selektif, sehingga lebih tahan terhadap masalah vanishing gradient[1]

### Arsitektur Model:

* Input shape: (24, num\_features)
* Dua LSTM layers:

  * Layer pertama: LSTM(100, activation='relu', return\_sequences=True)
  * Layer kedua: LSTM(50, activation='relu')
* Dropout layer:

  * Dropout(0.2) setelah masing-masing layer LSTM untuk mengurangi overfitting.
* Dense layer:

  * Dense(1) sebagai output untuk memprediksi `traffic_volume`
* Optimizer: Adam (default learning\_rate=0.001)
* Loss Function: Mean Squared Error (MSE)

## Evaluation

### Metrik Evaluasi:

* **MAE (Mean Absolute Error):** rata-rata kesalahan absolut.
* **RMSE (Root Mean Squared Error):** penalti lebih besar untuk kesalahan besar.
* **MAPE (Mean Absolute Percentage Error):** error relatif dalam persen.

### Hasil Evaluasi Model Multivariate:

| Metrik | Nilai  |
| ------ | ------ |
| MAE    | 245.55 |
| RMSE   | 366.85 |
| MAPE   | 12.56% |

### Visualisasi:

* Grafik prediksi vs aktual menunjukkan model mengikuti pola aktual dengan cukup baik, terutama saat volume tinggi di jam sibuk.
* Prediksi cenderung sedikit lagging terhadap lonjakan mendadak (seperti hari libur atau cuaca ekstrem).

### Analisis:

* LSTM multivariate memberikan performa lebih baik dari univariate, terutama pada jam sibuk dan hari libur.
* Fitur waktu (hour, day) dan `weather_main` memiliki kontribusi penting terhadap akurasi prediksi.

### Relevansi Terhadap Masalah:

Model ini dapat digunakan oleh pengelola transportasi untuk:

* Mengatur jadwal buka tutup jalur alternatif.
* Memberikan prediksi lalu lintas kepada pengguna melalui aplikasi navigasi.
* Mengantisipasi lonjakan lalu lintas saat cuaca buruk atau liburan.

## Kesimpulan

Proyek ini berhasil membangun model prediksi volume lalu lintas menggunakan pendekatan LSTM dengan performa yang cukup baik. Dengan memanfaatkan fitur waktu dan cuaca serta memproses data dengan teknik yang tepat, LSTM mampu menangkap pola-pola penting dalam data historis. Model ini dapat menjadi dasar untuk sistem pendukung keputusan dalam manajemen lalu lintas perkotaan.

### Saran Pengembangan:

* Menjelajahi pendekatan lain seperti Transformer atau hybrid model (CNN-LSTM).
* Mengembangkan sistem prediksi real-time berbasis API lalu lintas.

## Referensi

\[1] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
\[2] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.
\[3] F. Mohammed, "Metro Interstate Traffic Volume", Kaggle, 2024. \[Online]. Available: [https://www.kaggle.com/datasets/ujjwalchowdhury/metro-interstate-traffic-volume](https://www.kaggle.com/datasets/fekihmea/metro-interstate-traffic-volume/data).
