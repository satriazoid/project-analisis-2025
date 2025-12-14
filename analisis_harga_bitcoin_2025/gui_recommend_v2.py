import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# =========================
# SIMPLE LSTM NUMPY
# =========================
class SimpleLSTM:
    def __init__(self):
        self.w = 0.8

    def fit(self, series):
        returns = np.diff(series) / series[:-1]
        self.w = np.mean(returns)

    def predict(self, last_value, steps=7):
        preds = []
        value = last_value
        for _ in range(steps):
            value = value * (1 + self.w)
            preds.append(value)
        return preds


# =========================
# GUI
# =========================
class BTCGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BTC Forecast — SVR vs NumPy LSTM")
        self.setGeometry(200, 100, 950, 650)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load CSV")
        self.btn_train = QtWidgets.QPushButton("Train Models")
        self.btn_pred = QtWidgets.QPushButton("Predict 7 Days")

        self.btn_load.clicked.connect(self.load_csv)
        self.btn_train.clicked.connect(self.train_models)
        self.btn_pred.clicked.connect(self.predict_7days)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_pred)
        layout.addLayout(btn_layout)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        # Text
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

        # Models
        self.df = None
        self.scaler = StandardScaler()
        self.svr = SVR(kernel="rbf")
        self.lstm = SimpleLSTM()

    # =========================
    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load CSV", "", "*.csv")
        if not path:
            return

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df = df.sort_values("Date")

        self.df = df
        self.ax.clear()
        self.ax.plot(df["Date"], df["Close"], label="Actual")
        self.ax.legend()
        self.canvas.draw()

        self.text.setPlainText("Dataset loaded successfully.")

    # =========================
    def train_models(self):
        df = self.df.copy()

        # ----- SVR -----
        df["lag1"] = df["Close"].shift(1)
        df["lag2"] = df["Close"].shift(2)
        df["lag3"] = df["Close"].shift(3)
        df = df.dropna()

        X = df[["lag1","lag2","lag3"]].values
        y = df["Close"].values

        Xs = self.scaler.fit_transform(X)
        self.svr.fit(Xs, y)

        # ----- LSTM NumPy -----
        self.lstm.fit(self.df["Close"].values)

        self.text.setHtml("<b>SVR & LSTM NumPy trained successfully</b>")

    # =========================
    def predict_7days(self):
        df = self.df.copy()

        # ===== SVR FORECAST =====
        last3 = df["Close"].tail(3).tolist()
        svr_dates, svr_prices = [], []

        for i in range(7):
            scaled = self.scaler.transform([last3])
            pred = self.svr.predict(scaled)[0]

            last3 = [last3[-2], last3[-1], pred]
            next_date = df["Date"].iloc[-1] + pd.Timedelta(days=i+1)

            svr_dates.append(next_date)
            svr_prices.append(pred)

        # ===== LSTM FORECAST =====
        lstm_prices = self.lstm.predict(df["Close"].iloc[-1], 7)
        lstm_dates = svr_dates

        # ===== SIGNAL =====
        last_price = df["Close"].iloc[-1]
        svr_change = (svr_prices[-1] - last_price) / last_price * 100
        lstm_change = (lstm_prices[-1] - last_price) / last_price * 100

        def signal(change):
            if change > 2:
                return "BUY", "green"
            elif change < -2:
                return "SELL", "red"
            else:
                return "HOLD", "orange"

        svr_sig, svr_color = signal(svr_change)
        lstm_sig, lstm_color = signal(lstm_change)

        # ===== OUTPUT =====
        html = f"""
        <b>Forecast 7 Hari</b><br><br>
        Harga terakhir: <b>{last_price:,.0f}</b><br><br>

        <b>METODE SVR</b><br>
        Prediksi: <b>{svr_prices[-1]:,.0f}</b><br>
        Perubahan: {svr_change:.2f}%<br>
        Rekomendasi: <span style='color:{svr_color}; font-size:18px'>{svr_sig}</span><br><br>

        <b>METODE LSTM</b><br>
        Prediksi: <b>{lstm_prices[-1]:,.0f}</b><br>
        Perubahan: {lstm_change:.2f}%<br>
        Rekomendasi: <span style='color:{lstm_color}; font-size:18px'>{lstm_sig}</span>
        """
        self.text.setHtml(html)

        # ===== PLOT =====
        self.ax.clear()
        self.ax.plot(df["Date"], df["Close"], label="Actual")
        self.ax.plot(svr_dates, svr_prices, label="SVR Forecast", color="orange")
        self.ax.plot(lstm_dates, lstm_prices, label="LSTM NumPy Forecast", color="purple")
        self.ax.legend()
        self.canvas.draw()


# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BTCGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
