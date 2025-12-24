import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =======================
#   FUZZY MEMBERSHIP
# =======================

def mf_rendah(x,a,b):
    return max(0, min(1, (b-x)/(b-a)))

def mf_tinggi(x,a,b):
    return max(0, min(1, (x-a)/(b-a)))

def fuzzy_input1(x):
    rendah  = mf_rendah(x, 0, 60)
    sedang  = max(0, min((x-50)/20, (80-x)/20)) if 50<=x<=80 else 0
    tinggi  = mf_tinggi(x, 70, 100)
    return rendah, sedang, tinggi

def fuzzy_input2(x):
    kurang = mf_rendah(x, 0, 40)
    baik = max(0, min((x-30)/30, (70-x)/30)) if 30<=x<=70 else 0
    sangat = mf_tinggi(x, 60, 100)
    return kurang, baik, sangat

def fuzzy_input3(x):
    kurang = mf_rendah(x, 0, 40)
    cukup = max(0, min((x-30)/30, (70-x)/30)) if 30<=x<=70 else 0
    baik = mf_tinggi(x, 60, 100)
    return kurang, cukup, baik

def fuzzy_input4(x):
    stabil_low = mf_rendah(x, 0, 50)
    stabil_high = max(0, min((x-40)/20, 1)) if 40<=x<=60 else 0
    progres = mf_tinggi(x, 60, 100)
    return stabil_low, stabil_high, progres


# =======================
#   RULE BASE MAMDANI
# =======================

def fuzzy_output(i1, i2, i3, i4):
    # Inisialisasi dulu
    rendah = 0
    sedang = 0
    tinggi = 0
    random = 0

    # ===== RULE BASE =====

    # Rule Rendah
    rendah = max(rendah, min(i1[0], i2[0], i3[0]))
    rendah = max(rendah, i4[0])  # stabil_low menguatkan rendah

    # Rule Sedang (lebih fleksibel)
    sedang = max(sedang, min(i1[1], i2[1]))
    sedang = max(sedang, min(i1[1], i4[1]))
    sedang = max(sedang, min(i2[1], i3[1]))
    sedang = max(sedang, i2[2] * 0.5)  # kalau prestasi sangat baik → minimal sedang

    # Rule Tinggi
    tinggi = max(tinggi, min(i1[2], i2[2]))
    tinggi = max(tinggi, min(i1[2], i4[2]))
    tinggi = max(tinggi, min(i2[2], i3[2]))
    tinggi = max(tinggi, min(i4[1], i4[2]))  # stabil tinggi + progres tinggi → tinggi

    # Rule Random (kasus anomali/outlier)
    random = max(random, min(i1[2], i2[1], i3[1], i4[0]))  # tinggi tapi dasar tidak stabil
    random = max(random, min(i1[1], i2[2], i3[0]))         # prestasi naik, data inkonsisten

    return {
        "Rendah": rendah,
        "Sedang": sedang,
        "Tinggi": tinggi,
        "Random": random
    }



# =======================
#   GUI REALTIME
# =======================

class FuzzyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Analisis Beasiswa - Fuzzy Realtime")
        self.resize(900, 600)

        # 🔥 Fix: gunakan layout utama yang tersimpan dalam variabel
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Sliders
        self.s1 = self.make_slider("Input 1 (Skala Faktor)")
        self.s2 = self.make_slider("Input 2 (Prestasi / Keaktifan)")
        self.s3 = self.make_slider("Input 3 (Konsistensi)")
        self.s4 = self.make_slider("Input 4 (Progres / Stabilitas)")

        # Output Label
        self.out = QLabel("Output: -")
        self.out.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(self.out)

        # Grafik
        self.fig = Figure(figsize=(4,3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.main_layout.addWidget(self.canvas)

        # Realtime signal
        for s in (self.s1, self.s2, self.s3, self.s4):
            s.valueChanged.connect(self.update_fuzzy)

        self.update_fuzzy()

    def make_slider(self, name):
        box = QVBoxLayout()
        label = QLabel(name)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        box.addWidget(label)
        box.addWidget(slider)

        # 🔥 Fix: tambahkan layout menggunakan main_layout, bukan self.layout()
        self.main_layout.addLayout(box)
        return slider


    def update_fuzzy(self):
        i1 = fuzzy_input1(self.s1.value())
        i2 = fuzzy_input2(self.s2.value())
        i3 = fuzzy_input3(self.s3.value())
        i4 = fuzzy_input4(self.s4.value())

        hasil = fuzzy_output(i1,i2,i3,i4)
        keputusan = max(hasil, key=hasil.get)

        self.out.setText(f"Output: {keputusan} (μ={hasil[keputusan]:.2f})")

        # Update Chart
        self.ax.clear()
        self.ax.bar(hasil.keys(), hasil.values())
        self.ax.set_ylim(0,1)
        self.ax.set_ylabel("Tingkat Keanggotaan (μ)")
        self.canvas.draw()


# =======================
#   START PROGRAM
# =======================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FuzzyGUI()
    win.show()
    sys.exit(app.exec_())
