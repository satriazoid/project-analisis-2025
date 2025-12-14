import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class BTCGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PREDIKSI ANALISIS - BITCOIN 2025")
        self.setGeometry(200, 100, 900, 650)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        v = QtWidgets.QVBoxLayout(widget)

        # ------------------------------
        #   BUTTON AREA
        # ------------------------------
        h = QtWidgets.QHBoxLayout()

        self.btn_load = QtWidgets.QPushButton("Load CSV")
        self.btn_load.clicked.connect(self.load_csv)
        h.addWidget(self.btn_load)

        self.btn_train = QtWidgets.QPushButton("Train Model (SVR)")
        self.btn_train.clicked.connect(self.train_model)
        h.addWidget(self.btn_train)

        self.btn_pred7 = QtWidgets.QPushButton("Predict 7 Days")
        self.btn_pred7.clicked.connect(self.predict_7days)
        h.addWidget(self.btn_pred7)

        v.addLayout(h)

        # ------------------------------
        #   PLOT AREA
        # ------------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas)

        # ------------------------------
        #   TEXT OUTPUT
        # ------------------------------
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        v.addWidget(self.text)

        # Internal vars
        self.df = None
        self.svr = None
        self.scaler = None


    # =========================================================
    #   LOAD CSV
    # =========================================================
    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load CSV", "", "*.csv")
        if not path:
            return

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df = df.sort_values("Date")

        self.df = df
        self.plot_price()

        self.text.setHtml("<b>Dataset berhasil dimuat.</b>")


    # =========================================================
    #   PLOT ACTUAL PRICE
    # =========================================================
    def plot_price(self, preds=None):
        self.ax.clear()
        self.ax.plot(self.df["Date"], self.df["Close"], label="Actual Price")

        if preds:
            dts, vals = preds
            self.ax.plot(dts, vals, label="Forecast", marker="o")

        self.ax.legend()
        self.canvas.draw()


    # =========================================================
    #   TRAIN SVR MODEL
    # =========================================================
    def train_model(self):
        df = self.df.copy()

        # Lag features
        df["lag1"] = df["Close"].shift(1)
        df["lag2"] = df["Close"].shift(2)
        df["lag3"] = df["Close"].shift(3)
        df = df.dropna()

        X = df[["lag1", "lag2", "lag3"]].values
        y = df["Close"].values

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.svr = SVR(kernel="rbf")
        self.svr.fit(Xs, y)
        close_series = self.df["Close"].values
        self.lstm.fit(close_series)
        self.text.setHtml("<b>Training SVR selesai!</b>")


    # =========================================================
    #   PREDICT 7 DAYS + SIGNAL
    # =========================================================
    def predict_7days(self):
        if self.svr is None:
            self.text.setHtml("<b>Silakan Train Model terlebih dahulu.</b>")
            return

        df = self.df.copy()

        # Gunakan 3 closing terakhir
        last3 = df["Close"].tail(3).tolist()

        future_dates = []
        future_prices = []

        for i in range(7):
            scaled = self.scaler.transform([last3])
            pred = float(self.svr.predict(scaled)[0])

            last3 = [last3[-2], last3[-1], pred]

            next_date = df["Date"].iloc[-1] + pd.Timedelta(days=i + 1)
            future_dates.append(next_date)
            future_prices.append(pred)

        # ------ SIGNAL SECTION ------
        last_price = df["Close"].iloc[-1]
        final_forecast = future_prices[-1]

        change_percent = ((final_forecast - last_price) / last_price) * 100

        if change_percent > 2:
            signal = "BUY"
            color = "green"
        elif change_percent > -2:
            signal = "SELL"
            color = "red"
        else:
            signal = "HOLD"
            color = "orange"

        # ------ WRITE TEXT ------
        html = f"""
        <br>Forecast 7 Hari:</br><br>
        Harga terakhir: <b>{last_price:,.0f}</b><br>
        Prediksi hari ke-7: <b>{final_forecast:,.0f}</b><br>
        Perubahan: <b>{change_percent:.2f}%</b><br><br>

        <b>Rekomendasi:</b>
        <span style='color:{color}; font-size:14px;'>{signal}</span>
        """

        self.text.setHtml(html)

        # ------ DRAW GRAPH ------
        self.ax.clear()
        self.ax.plot(df["Date"], df["Close"], label="Actual Price")
        self.ax.plot(future_dates, future_prices, label="Forecast 7 Hari", color="orange")
        self.ax.scatter(future_dates[-1], future_prices[-1], color=color, s=150)
        self.ax.legend()
        self.canvas.draw()


# =========================================================
#   MAIN
# =========================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BTCGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()