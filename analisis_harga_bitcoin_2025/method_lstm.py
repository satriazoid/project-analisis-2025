# method_lstm.py
# Usage: python method_lstm.py
# Expects processed CSV. Output: model_lstm.h5 and results_lstm.json

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    p = Path("dataset_btc_processed.csv")
    df = pd.read_csv(p, parse_dates=['Date'])
    return df

def create_sequences(values, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i])
    return np.array(X), np.array(y)

def train_lstm():
    df = load_data()
    close = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    close_s = scaler.fit_transform(close)

    seq_len = 30
    X, y = create_sequences(close_s.flatten(), seq_len=seq_len)
    # reshape X to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test), callbacks=[es], verbose=2)

    y_pred_s = model.predict(X_test).flatten()
    # inverse scale
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_s.reshape(-1,1)).flatten()

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

    model.save("model_lstm.h5")
    # save scaler
    import joblib
    joblib.dump({'scaler': scaler, 'seq_len': seq_len}, "lstm_scaler.pkl")

    results = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape), 'n_train':len(X_train), 'n_test':len(X_test)}
    with open("results_lstm.json","w") as f:
        json.dump(results, f, indent=2)
    print("LSTM results:", results)
    return results

if __name__ == "__main__":
    train_lstm()
