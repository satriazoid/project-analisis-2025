import pandas as pd
import numpy as np
import os

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# --- 1. Fungsi Preprocessing (Tahap 3.4.1) ---
def create_time_series_window(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- 2. Fungsi Evaluasi (Tahap 3.4.4) ---
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# --- 3. Proses Utama LSTM ---
def run_lstm_analysis(file_path):
    # a. Load Data & Cleansing
    df = pd.read_csv('dataset_btc.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Pilih kolom yang akan diprediksi (misalnya 'Close')
    data = df['Close'].values.reshape(-1, 1)

    # b. Normalisasi (Min-Max Scaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # c. Pembagian Data (Train/Test Split)
    # Gunakan time-series split (bukan random)
    training_data_len = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :] # Ambil 60 hari terakhir sebagai look_back untuk data uji
    
    # d. Pembentukan Window Time Series (Time Series Windowing)
    look_back = 60 # Jendela waktu input (misalnya 60 hari)
    X_train, y_train = create_time_series_window(train_data, look_back)
    X_test, y_test = create_time_series_window(test_data, look_back)
    
    # Reshape data untuk LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # e. Perancangan & Pelatihan Model LSTM (Tahap 3.4.2 & 3.4.3)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1) # Sesuaikan epochs untuk pelatihan sebenarnya

    # f. Prediksi (Tahap 3.4.4)
    predictions_scaled = model.predict(X_test)
    
    # g. Inverse Transform (Mengembalikan ke skala harga asli)
    # Perlu data 'y_test' asli yang belum di-reshape untuk inversi
    # Kita hanya perlu hasil prediksi untuk membalik skala
    dummy_array = np.zeros(shape=(len(predictions_scaled), data.shape[1]))
    dummy_array[:,0] = predictions_scaled[:,0]
    predictions = scaler.inverse_transform(dummy_array)[:,0]

    # Ambil nilai aktual (y_test) untuk perbandingan
    # Kita perlu mengambil y_test yang telah di-inverse_transform
    y_test_original = data[training_data_len + look_back + 1:]
    
    # h. Evaluasi
    mae, rmse, mape = evaluate_model(y_test_original, predictions)
    
    # Hasil
    results = {
        'Model': 'LSTM',
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    return results, y_test_original, predictions

# # TODO: Ganti dengan path file CSV data Bitcoin Anda
# file_path = 'data_bitcoin_2020_2025.csv'
# results_lstm, actual_lstm, pred_lstm = run_lstm_analysis(file_path)
# print("Hasil Analisis LSTM:", results_lstm)