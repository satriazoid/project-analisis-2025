# gui_recommend.py
# Minimalist Qt GUI: predict next-day BTC price using SVR and LSTM, give recommendation, and show checklist.
# Usage: python gui_recommend.py
# Requirements: PyQt5, pandas, numpy, scikit-learn, joblib, matplotlib, tensorflow

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf

from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ML libs
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Try to import tensorflow, but handle if not available
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------------
# Helper ML functions
# -------------------------
def train_svr_from_df(df, save_path="model_svr.pkl"):
    # features: lag_1..lag_N and roll_*
    feature_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]
    X = df[feature_cols].values
    y = df['Close'].values
    split = int(len(df) * 0.8) or 1
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svr = SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale')
    svr.fit(X_train_s, y_train)
    # save
    joblib.dump({'model': svr, 'scaler': scaler, 'features': feature_cols}, save_path)
    # eval quick
    y_pred = svr.predict(X_test_s) if len(X_test_s)>0 else []
    res = {}
    if len(y_pred)>0:
        res['mae'] = float(mean_absolute_error(y_test, y_pred))
        res['rmse'] = float(sqrt(mean_squared_error(y_test, y_pred)))
    else:
        res['mae'] = None; res['rmse'] = None
    return save_path, res

def train_lstm_from_df(df, save_path="model_lstm.h5", scaler_path="lstm_scaler.pkl", seq_len=14, epochs=30):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available in environment.")
    close = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    close_s = scaler.fit_transform(close)
    # create sequences
    X, y = [], []
    for i in range(seq_len, len(close_s)):
        X.append(close_s[i-seq_len:i, 0])
        y.append(close_s[i, 0])
    X = np.array(X)
    y = np.array(y)
    if len(X)==0:
        raise ValueError("Not enough data for LSTM sequence length " + str(seq_len))
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # --- KODE PERBAIKAN UNTUK PEMBAGIAN DATA DIMULAI ---
    N = len(X)
    if N <= 5:
        # Jika data sangat sedikit, gunakan 100% untuk training, dan lewati evaluasi
        X_train, X_test = X, np.array([])
        y_train, y_test = y, np.array([])
        # Peringatan: Model dilatih pada 100% data, validasi MAE/RMSE akan None.
    else:
        # Bagi 80/20 secara normal
        split = int(N * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    # --- KODE PERBAIKAN SELESAI ---
    # small model
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.15),
        LSTM(16),
        Dropout(0.15),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_test, y_test) if len(X_test) > 0 else None, verbose=0)
    model.save(save_path)
    joblib.dump({'scaler': scaler, 'seq_len': seq_len}, scaler_path)
    # eval quick
    y_pred = model.predict(X_test).flatten() if len(X_test)>0 else []
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten() if len(y_test)>0 else []
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten() if len(y_pred)>0 else []
    res = {}
    if len(y_pred_inv)>0:
        res['mae'] = float(mean_absolute_error(y_test_inv, y_pred_inv))
        res['rmse'] = float(sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    else:
        res['mae'] = None; res['rmse'] = None
    return save_path, scaler_path, res

def predict_next_svr(df, model_path="model_svr.pkl"):
    # expects model saved by train_svr_from_df
    rec = joblib.load(model_path)
    model = rec['model']; scaler = rec['scaler']; features = rec['features']
    # take last row of df for features
    x = df[features].iloc[-1].values.reshape(1,-1)
    x_s = scaler.transform(x)
    pred = model.predict(x_s)[0]
    return pred

def predict_next_lstm(df, model_path="model_lstm.h5", scaler_path="lstm_scaler.pkl"):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")
    rec = joblib.load(scaler_path)
    scaler = rec['scaler']; seq_len = rec['seq_len']
    model = load_model(model_path)
    close = df['Close'].values.reshape(-1,1)
    close_s = scaler.transform(close)
    if len(close_s) < seq_len:
        raise ValueError("Not enough data to predict with LSTM. seq_len=" + str(seq_len))
    last_seq = close_s[-seq_len:,0].reshape(1, seq_len, 1)
    pred_s = model.predict(last_seq).flatten()[0]
    pred = scaler.inverse_transform(np.array([[pred_s]]))[0,0]
    return pred

# -------------------------
# Recommendation logic
# -------------------------
def simple_recommendation(last_price, predicted_price, threshold_pct=0.3):
    """
    Basic rules:
    - If predicted_price >= last_price * (1 + threshold_pct/100) -> Buy
    - If predicted_price <= last_price * (1 - threshold_pct/100) -> Sell
    - Else -> Hold
    threshold_pct default 0.3% (can be adjusted)
    """
    if last_price <= 0:
        return "No data"
    change_pct = (predicted_price - last_price) / last_price * 100
    if change_pct >= threshold_pct:
        return f"BUY (predicted +{change_pct:.2f}%)"
    elif change_pct <= -threshold_pct:
        return f"SELL (predicted {change_pct:.2f}%)"
    else:
        return f"HOLD (predicted {change_pct:.2f}%)"

# -------------------------
# GUI
# -------------------------
class BTCAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC Recommender (LSTM & SVR)")
        self.setGeometry(150, 100, 1000, 700)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        v = QtWidgets.QVBoxLayout(widget)

        # Top controls
        h = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load processed CSV")
        self.btn_load.clicked.connect(self.load_csv)
        h.addWidget(self.btn_load)

        self.btn_train = QtWidgets.QPushButton("Train Models (quick)")
        self.btn_train.clicked.connect(self.train_models)
        h.addWidget(self.btn_train)

        self.btn_predict = QtWidgets.QPushButton("Predict & Recommend")
        self.btn_predict.clicked.connect(self.predict_and_recommend)
        h.addWidget(self.btn_predict)

        self.seq_len_spin = QtWidgets.QSpinBox(); self.seq_len_spin.setRange(5,60); self.seq_len_spin.setValue(14)
        h.addWidget(QtWidgets.QLabel("LSTM seq_len:"))
        h.addWidget(self.seq_len_spin)

        self.threshold_spin = QtWidgets.QDoubleSpinBox(); self.threshold_spin.setRange(0.0,10.0); self.threshold_spin.setValue(0.3)
        self.threshold_spin.setSuffix(" %"); self.threshold_spin.setSingleStep(0.1)
        h.addWidget(QtWidgets.QLabel("Decision threshold:"))
        h.addWidget(self.threshold_spin)

        v.addLayout(h)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(9,4))
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)

        # Results area
        lower = QtWidgets.QHBoxLayout()

        self.txt_results = QtWidgets.QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setFixedWidth(420)
        lower.addWidget(self.txt_results)

        # Checklist / guidance
        self.chk = QtWidgets.QTextEdit()
        self.chk.setReadOnly(True)
        self.chk.setPlainText(self._default_checklist_text())
        lower.addWidget(self.chk)

        v.addLayout(lower)

        # status bar
        self.status = QtWidgets.QLabel("Ready")
        v.addWidget(self.status)

        # internal state
        self.df = None
        self.model_svr_path = "model_svr.pkl"
        self.model_lstm_path = "model_lstm.h5"
        self.lstm_scaler_path = "lstm_scaler.pkl"

    def _default_checklist_text(self):
        return (
            "Checklist — Hal yang perlu dicek secara berkala saat memprediksi harga:\n\n"
            "1. Data freshness: pastikan CSV terbaru (tanggal & waktu) — gunakan data harian/menit sesuai kebutuhan.\n"
            "2. Volume & liquidity: periksa 'Volume' jika tersedia; perubahan volume dapat menandakan kekuatan tren.\n"
            "3. Peristiwa makro: berita besar, halving, regulasi—catat tanggal kejadian.\n"
            "4. Validasi model: pantau MAE/RMSE secara berkala setelah retraining.\n"
            "5. Window/lag sensitivity: jika dataset kecil turunkan seq_len/lag.\n"
            "6. Overfitting: cek gap antara train/val loss pada LSTM.\n"
            "7. Ensemble: bandingkan prediksi beberapa model, gunakan consensus.\n"
            "8. Risk management: tetapkan stop-loss / take-profit sebelum trading.\n"
            "\n(Scroll untuk tips lebih lanjut.)"
        )

    def load_csv(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open processed CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not p:
            return
        try:
            df = pd.read_csv(p, parse_dates=['Date'])
        except Exception:
            df = pd.read_csv(p)
        # Expect at least Date and Close
        if 'Date' not in df.columns or 'Close' not in df.columns:
            QtWidgets.QMessageBox.critical(self, "Error", "CSV harus memiliki kolom 'Date' dan 'Close'.")
            return
        self.df = df.sort_values('Date').reset_index(drop=True)
        self.status.setText(f"Loaded {len(self.df)} rows from {os.path.basename(p)}")
        self.plot_price()

    def plot_price(self, preds=None):
        self.ax.clear()
        self.ax.plot(self.df['Date'], self.df['Close'], label='Close')
        if preds is not None:
            # preds: list of tuples (date, value, label)
            for d, v, lab in preds:
                self.ax.scatter([d], [v], label=lab, marker='X', s=80)
        self.ax.legend()
        self.ax.set_title("BTC Close Price")
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def train_models(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load processed CSV first.")
            return
        
        # Ensure we have features for SVR: if not, create them and STORE to self.df
        df = self.df.copy()

        max_lag = 14
        # buat lag
        for lag in range(1, max_lag + 1):
            col = f'lag_{lag}'
            if col not in df.columns:
                df[col] = df['Close'].shift(lag)

        # buat rolling mean
        for w in [7, 14]:
            col = f'roll_mean_{w}'
            if col not in df.columns:
                df[col] = df['Close'].rolling(window=w).mean()

        # hapus NaN
        df = df.dropna().reset_index(drop=True)

        # SIMPAN KEMBALI agar dipakai saat prediction
        self.df = df.copy()

        # Train SVR (quick)
        self.status.setText("Training SVR...")
        QtWidgets.QApplication.processEvents()
        try:
            path_svr, res_svr = train_svr_from_df(df, save_path=self.model_svr_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SVR error", str(e))
            return

        # Train LSTM if available
        lstm_res = None
        if TF_AVAILABLE:
            try:
                seq_len = int(self.seq_len_spin.value())
                self.status.setText("Training LSTM (may take a bit)...")
                QtWidgets.QApplication.processEvents()
                path_lstm, path_scaler, res_lstm = train_lstm_from_df(df, save_path=self.model_lstm_path, scaler_path=self.lstm_scaler_path, seq_len=seq_len, epochs=25)
                lstm_res = res_lstm
            except Exception as e:
                lstm_res = {'error': str(e)}
        else:
            lstm_res = {'error': 'TensorFlow not available'}

        # Save a quick results file
        summary = {'svr': res_svr, 'lstm': lstm_res}
        with open("results_quick_train.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.status.setText("Training finished.")
        self.txt_results.setPlainText("Training finished.\n\nSVR eval: {}\n\nLSTM eval: {}\n\nModels saved.".format(res_svr, lstm_res))

    def predict_and_recommend(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load processed CSV first.")
            return
        last_price = float(self.df['Close'].iloc[-1])
        results_text = []
        preds = []
        threshold = float(self.threshold_spin.value())

        # SVR predict
        if os.path.exists(self.model_svr_path):
            try:
                pred_svr = predict_next_svr(self.df, model_path=self.model_svr_path)
                rec_svr = simple_recommendation(last_price, pred_svr, threshold_pct=threshold)
                results_text.append(f"SVR prediction: {pred_svr:,.2f}\nRecommendation: {rec_svr}\n")
                # assume next "date" is last_date + 1 day (not exact for intraday)
                next_date = pd.to_datetime(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1)
                preds.append((next_date, pred_svr, "SVR"))
            except Exception as e:
                results_text.append("SVR prediction error: " + str(e))
        else:
            results_text.append("SVR model not found. Train models first (click Train Models).")

        # LSTM predict
        if os.path.exists(self.model_lstm_path) and os.path.exists(self.lstm_scaler_path) and TF_AVAILABLE:
            try:
                pred_lstm = predict_next_lstm(self.df, model_path=self.model_lstm_path, scaler_path=self.lstm_scaler_path)
                rec_lstm = simple_recommendation(last_price, pred_lstm, threshold_pct=threshold)
                results_text.append(f"LSTM prediction: {pred_lstm:,.2f}\nRecommendation: {rec_lstm}\n")
                next_date = pd.to_datetime(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1)
                preds.append((next_date, pred_lstm, "LSTM"))
            except Exception as e:
                results_text.append("LSTM prediction error: " + str(e))
        else:
            if not TF_AVAILABLE:
                results_text.append("LSTM not available (TensorFlow missing).")
            else:
                results_text.append("LSTM model or scaler not found. Train models first (click Train Models).")

        # Quick simple "historical check" (yesterday up/down)
        if len(self.df) >= 2:
            yesterday = float(self.df['Close'].iloc[-2])
            today = float(self.df['Close'].iloc[-1])
            change_pct = (today - yesterday) / yesterday * 100 if yesterday!=0 else 0.0
            hist = f"Historical (yesterday -> today): {yesterday:,.2f} -> {today:,.2f} ({change_pct:.2f}%)"
        else:
            hist = "Not enough history to compare yesterday."

        # Consensus recommendation (simple)
        recs = []
        for line in results_text:
            if "BUY" in line:
                recs.append("BUY")
            elif "SELL" in line:
                recs.append("SELL")
            elif "HOLD" in line:
                recs.append("HOLD")
        if recs:
            # majority voting, else HOLD
            from collections import Counter
            cnt = Counter(recs)
            consensus = cnt.most_common(1)[0][0]
            consensus_text = f"Consensus (majority): {consensus}"
        else:
            consensus_text = "Consensus: N/A"

        # Display
        out_text = "Last price: {:,.2f}\n\n".format(last_price)
        out_text += hist + "\n\n"
        out_text += "\n".join(results_text) + "\n"
        out_text += consensus_text + "\n"
        out_text += "\nNote: Recommendations are simplistic rules based on predicted price change and a threshold (adjustable)."
        self.txt_results.setPlainText(out_text)

        # plot with preds
        if preds:
            self.plot_price(preds=preds)
        else:
            self.plot_price()

# -------------------------
# Run app
# -------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BTCAnalyzer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
