import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # SVR sensitif skala, StandardScaler lebih umum
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# --- 1. Fungsi Feature Engineering (Lag Features) ---
def create_lag_features(df, lag=60, target_col='Close'):
    df_lag = df[[target_col]].copy()
    for i in range(1, lag + 1):
        df_lag[f'Lag_{i}'] = df_lag[target_col].shift(i)
    df_lag.dropna(inplace=True)
    
    # X adalah fitur (harga penutupan dari 60 hari sebelumnya)
    X = df_lag.drop(columns=[target_col]).values
    # Y adalah target (harga penutupan hari ini)
    y = df_lag[target_col].values
    return X, y

# --- 2. Fungsi Evaluasi (Tahap 3.4.4) ---
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# --- 3. Proses Utama SVR ---
def run_svr_analysis(file_path):
    # a. Load Data & Cleansing
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # b. Feature Engineering (Lag)
    lag_days = 60 # Jumlah hari lag
    X_full, y_full = create_lag_features(df, lag=lag_days, target_col='Close')

    # c. Pembagian Data (Train/Test Split)
    training_data_len = math.ceil(len(X_full) * 0.8)
    X_train, y_train = X_full[:training_data_len], y_full[:training_data_len]
    X_test, y_test = X_full[training_data_len:], y_full[training_data_len:]

    # d. Standardisasi (Preprocessing)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # e. Pelatihan Model SVR (Tahap 3.4.3)
    # Gunakan Kernel RBF (paling umum)
    # TODO: Parameter C, epsilon, dan gamma bisa di-tuning (grid search)
    model = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1) 
    model.fit(X_train_scaled, y_train)

    # f. Prediksi (Tahap 3.4.4)
    predictions = model.predict(X_test_scaled)
    
    # g. Evaluasi
    mae, rmse, mape = evaluate_model(y_test, predictions)
    
    # Hasil
    results = {
        'Model': 'SVR',
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    return results, y_test, predictions

# # TODO: Ganti dengan path file CSV data Bitcoin Anda
# file_path = 'data_bitcoin_2020_2025.csv'
# results_svr, actual_svr, pred_svr = run_svr_analysis(file_path)
# print("Hasil Analisis SVR:", results_svr)