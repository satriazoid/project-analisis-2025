import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate
import locale
import matplotlib.pyplot as plt # Import Matplotlib

# --- 1-4. (Input, Pre-processing, Pelatihan Model - Sama seperti sebelumnya) ---

# Data Placeholder
data_dict = {'desember-2024': [32080568, 541565582, 481519430, 19922231, 255137],
            'maret-2025': [29709278, 542666796, 441561126, 19901284, 2745731]}
index_labels = ['kas', 'tabungan', 'deposito', 'keuntungan', 'kerugian']
df_raw = pd.DataFrame(data_dict, index=index_labels)
df = df_raw.T 

Y = df['keuntungan']
X = df.drop('keuntungan', axis=1)
X_train, Y_train = X, Y

# Scaling Wajib untuk SVR
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train) 
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1)).flatten() 

svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1) 
svr_model.fit(X_train_scaled, Y_train_scaled)
Y_pred_scaled_svr = svr_model.predict(X_train_scaled)
Y_pred_svr = scaler_Y.inverse_transform(Y_pred_scaled_svr.reshape(-1, 1)).flatten()
rmse_svr = np.sqrt(mean_squared_error(Y_train, Y_pred_svr))
mae_svr = mean_absolute_error(Y_train, Y_pred_svr)

# Konfigurasi Locale untuk format Rupiah
try:
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8') 
except:
    locale.setlocale(locale.LC_ALL, 'Indonesian_Indonesia')

def format_rupiah(x):
    return locale.format_string("%d", x, grouping=True)


# --- 5. MENAMPILKAN HASIL & ANALISIS ---
print("--- 5. HASIL & ANALISIS SVR ---")

# a. Perbandingan Prediksi (Tabel)
hasil_df = pd.DataFrame({
    'Periode': df.index,
    'Keuntungan Aktual': Y_train,
    'Keuntungan Prediksi SVR': Y_pred_svr.round(0)
})

hasil_df['Keuntungan Aktual'] = hasil_df['Keuntungan Aktual'].apply(format_rupiah)
hasil_df['Keuntungan Prediksi SVR'] = hasil_df['Keuntungan Prediksi SVR'].apply(format_rupiah)

print("a. TABEL PERBANDINGAN AKTUAL vs. PREDIKSI:")
print(tabulate(hasil_df, headers='keys', tablefmt='fancy_grid', numalign="right"))

# b. Metrik Evaluasi (Tabel)
metrik_svr = pd.DataFrame({
    'Metrik': ['RMSE', 'MAE'],
    'Nilai': [round(rmse_svr, 0), round(mae_svr, 0)]
})
metrik_svr['Nilai'] = metrik_svr['Nilai'].apply(format_rupiah)

print("\nb. TABEL METRIK EVALUASI:")
print(tabulate(metrik_svr, headers='keys', tablefmt='fancy_grid', numalign="right"))

# --- 6. VISUALISASI HASIL (GRAFIK) ---

# 6.1. GRAFIK PERBANDINGAN PREDISI AKTUAL vs. SVR
hasil_df_plot = hasil_df.set_index('Periode')
# Ubah kolom yang sudah diformat Rupiah menjadi numerik untuk plotting
hasil_df_plot['Keuntungan Aktual'] = hasil_df_plot['Keuntungan Aktual'].str.replace('.', '', regex=False).astype(int)
hasil_df_plot['Keuntungan Prediksi SVR'] = hasil_df_plot['Keuntungan Prediksi SVR'].str.replace('.', '', regex=False).astype(int)

hasil_df_plot[['Keuntungan Aktual', 'Keuntungan Prediksi SVR']].plot(kind='bar', figsize=(8, 5))
plt.title('Perbandingan Keuntungan Aktual vs. Prediksi SVR')
plt.ylabel('Nilai Keuntungan')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.legend(loc='upper right')
plt.show()

# --- 7. Hasil Akhir dan Perbandingan (Koreksi Baris RMSE/MAE) ---

print("--- Hasil Prediksi dan Metrik Evaluasi SVR ---")

hasil_df = pd.DataFrame({
    'Periode': df.index,
    'Keuntungan Aktual': Y_train,
    'Keuntungan Prediksi SVR': Y_pred_svr.round(0)
})

hasil_df['Keuntungan Aktual'] = hasil_df['Keuntungan Aktual'].apply(format_rupiah)
hasil_df['Keuntungan Prediksi SVR'] = hasil_df['Keuntungan Prediksi SVR'].apply(format_rupiah)

print(tabulate(hasil_df, headers='keys', tablefmt='fancy_grid', numalign="right"))

# KOREKSI: Menggunakan fungsi round() bawaan Python
print(f"\nRMSE (Root Mean Squared Error): {format_rupiah(round(rmse_svr, 0))} Rupiah")
print(f"MAE (Mean Absolute Error): {format_rupiah(round(mae_svr, 0))} Rupiah")
print("-----------------------------------------------------")