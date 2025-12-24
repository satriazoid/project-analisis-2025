import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================
# DOMAIN GLOBAL
# ==============================
X = np.linspace(0,100,200)

# ==============================
# MEMBERSHIP FUNCTIONS
# ==============================
def trap(x,a,b,c,d):
    if b-a == 0: b = a + 1e-6
    if d-c == 0: c = d - 1e-6
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a), 1), (d-x)/(d-c)))

def trimf(x,a,b,c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))

# INPUT MEMBERSHIP
def mf_input1(v): return trap(v,0,0,30,50), trimf(v,40,60,80), trap(v,60,80,100,100)
def mf_input2(v): return trap(v,0,0,20,40), trimf(v,30,50,70), trap(v,60,80,100,100)
def mf_input3(v): return trap(v,0,0,20,40), trimf(v,30,50,70), trap(v,60,80,100,100)
def mf_input4(v): return trap(v,0,0,20,50), trimf(v,40,60,80), trap(v,60,80,100,100)

# ==============================
# RULE BASE (3 OUTPUT: Gagal/Digantung/Diterima)
# ==============================
def fuzzy_infer(i1, i2, i3, i4):
    gagal = max(min(i1[0], i2[0], i3[0]), i4[0])
    digantung = max(min(i1[1], i2[1]), min(i2[1], i3[1]), min(i1[1], i4[1]))
    diterima = max(min(i1[2], i2[2]), min(i2[2], i3[2]), min(i1[2], i4[2]))
    return {"Gagal":gagal, "Digantung":digantung, "Diterima":diterima}

# ==============================
# DEFUZZYFIKASI (CENTROID)
# ==============================
def defuzzyfikasi(hasil):
    z = np.linspace(0,100,200)

    μ_gagal     = np.where(z<=40, 1, np.where(z>=60,0,(60-z)/20)) * hasil["Gagal"]
    μ_digantung = np.where(z<=40, 0, np.where(z>=70,1,(z-40)/30)) * hasil["Digantung"]
    μ_diterima  = np.where(z<=60, 0, np.where(z>=80,1,(z-60)/20)) * hasil["Diterima"]

    μ_total = μ_gagal + μ_digantung + μ_diterima
    if μ_total.sum() == 0:
        return 0

    return (z * μ_total).sum() / μ_total.sum()

# ==============================
# GUI
# ==============================
class FuzzyManual(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seleksi Beasiswa - Analisis Fuzzy (Minimalist UI)")
        self.setGeometry(200,100,950,780)
        self.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                font-family: Arial;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                font-size: 14px;
                background: white;
            }
            QPushButton {
                background-color: #007BFF;
                color: white;
                padding: 10px;
                border-radius: 6px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QLabel {
                font-size: 14px;
            }
        """)

        # --- Layout Utama ---
        layout = QVBoxLayout()

        card = QGroupBox("🔍 Masukkan Parameter Seleksi Beasiswa")
        card.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; padding: 10px; }")
        form = QFormLayout()

        # --- Input dengan deskripsi ---
        self.in1 = QLineEdit(); self.in1.setPlaceholderText("contoh: 10 - 80")
        self.in2 = QLineEdit(); self.in2.setPlaceholderText("contoh: 30 - 90")
        self.in3 = QLineEdit(); self.in3.setPlaceholderText("contoh: 20 - 100")
        self.in4 = QLineEdit(); self.in4.setPlaceholderText("contoh: 40 - 100")

        form.addRow("Input 1 (Skala Faktor 10–80):", self.in1)
        form.addRow("Input 2 (Prestasi 30–90):", self.in2)
        form.addRow("Input 3 (Konsistensi 20–100):", self.in3)
        form.addRow("Input 4 (Progres/Stabilitas 40–100):", self.in4)

        card.setLayout(form)
        layout.addWidget(card)

        # Tombol eksekusi
        btn = QPushButton("HITUNG ANALISIS BEASISWA")
        btn.clicked.connect(self.proses)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        # Output Text
        self.output_label = QLabel("\nStatus: -")
        self.output_label.setStyleSheet("font-size: 22px; font-weight:bold; color:#0A4D68;")
        layout.addWidget(self.output_label, alignment=Qt.AlignCenter)

        self.defuz_label = QLabel("Nilai Defuzzyfikasi: -")
        self.defuz_label.setStyleSheet("font-size: 18px; font-weight:bold; color:#B31312;")
        layout.addWidget(self.defuz_label, alignment=Qt.AlignCenter)

        # Canvas Grafik
        self.fig = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.setLayout(layout)


    def proses(self):
        try:
            v1 = int(self.in1.text()); v2 = int(self.in2.text())
            v3 = int(self.in3.text()); v4 = int(self.in4.text())
        except:
            self.output_label.setText("⚠ Masukkan angka 0-100 dengan benar!")
            return

        i1,i2,i3,i4 = mf_input1(v1), mf_input2(v2), mf_input3(v3), mf_input4(v4)
        hasil = fuzzy_infer(i1,i2,i3,i4)
        keputusan = max(hasil, key=hasil.get)

        # Tampilkan keputusan fuzzy
        self.output_label.setText(f"Status Beasiswa: {keputusan.upper()} (μ={hasil[keputusan]:.2f})")

        # Hitung defuzzyfikasi
        nilai = defuzzyfikasi(hasil)
        self.defuz_label.setText(f"Nilai Defuzzyfikasi: {nilai:.2f} / 100")

        # Gambar kurva output
        self.ax.clear()
        self.ax.plot(X, trap(X,0,0,40,60), "r", label="Gagal")
        self.ax.plot(X, trimf(X,40,60,70), "orange", label="Digantung")
        self.ax.plot(X, trap(X,60,80,100,100), "g", label="Diterima")
        self.ax.set_title("Kurva Output Keputusan")
        self.ax.set_ylim(0,1.1)
        self.ax.legend()
        self.canvas.draw()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FuzzyManual()
    win.show()
    sys.exit(app.exec_())
