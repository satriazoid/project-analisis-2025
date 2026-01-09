# method_svr.py
# Usage: python method_svr.py
# Expects: dataset_btc_processed.csv in current folder (created by preprocess.py)
# Output: model_svr.pkl and metrics printed & saved to results_svr.json

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib, json
from pathlib import Path
from math import sqrt

def load_data():
    p = Path("dataset_btc_processed.csv")
    df = pd.read_csv(p, parse_dates=['Date'])
    return df

def train_and_eval():
    df = load_data()
    # use lag features as X
    feature_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]
    X = df[feature_cols].values
    y = df['Close'].values

    # Train/test split: last 20% as test (time-series split)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svr = SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale')
    svr.fit(X_train_s, y_train)

    y_pred = svr.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # save model + scaler
    joblib.dump({'model': svr, 'scaler': scaler, 'features': feature_cols}, "model_svr.pkl")

    results = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape),
               'n_train': len(X_train), 'n_test': len(X_test)}
    with open("results_svr.json", "w") as f:
        json.dump(results, f, indent=2)
    print("SVR results:", results)
    return results

if __name__ == "__main__":
    train_and_eval()
