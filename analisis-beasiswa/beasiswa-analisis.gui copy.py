import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

X = np.linspace(0,100,200)

# ==============================
# MEMBERSHIP FUNCTIONS
# ==============================

def trap(x,a,b,c,d):
    # Hindari pembagian 0
    if b == a:
        left = np.where(x <= b, 1, np.maximum(0, np.minimum((d-x)/(d-c), 1)))
    else:
        left = np.minimum((x-a)/(b-a), 1)

    if d == c:
        right = np.where(x >= c, 1, np.maximum(0, np.minimum((x-a)/(b-a), 1)))
    else:
        right = np.minimum((d-x)/(d-c), 1)

    return np.maximum(0, np.minimum(left, right))


def trimf(x,a,b,c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))

# Membership Input Domain

def mf_input1(x):
    rendah = trap(x,0,0,30,50)
    sedang = trimf(x,40,60,80)
    tinggi = trap(x,60,80,100,100)
    return rendah, sedang, tinggi

def mf_input2(x):
    kurang = trap(x,0,0,20,40)
    baik = trimf(x,30,50,70)
    sangat = trap(x,60,80,100,100)
    return kurang, baik, sangat

def mf_input3(x):
    kurang = trap(x,0,0,20,40)
    cukup = trimf(x,30,50,70)
    baik = trap(x,60,80,100,100)
    return kurang, cukup, baik

def mf_input4(x):
    stabil_low  = trap(x,0,0,20,50)
    stabil_high = trimf(x,40,60,80)
    progres = trap(x,60,80,100,100)
    return stabil_low, stabil_high, progres


# ==============================
# RULE BASE
# ==============================

def fuzzy_infer(i1, i2, i3, i4):
    gagal = max(min(i1[0], i2[0], i3[0]), i4[0])
    digantung = max(min(i1[1], i2[1]), min(i2[1], i3[1]), min(i1[1], i4[1]))
    diterima = max(min(i1[2], i2[2]), min(i2[2], i3[2]), min(i1[2], i4[2]))
    
    return {
        "Gagal": gagal,
        "Digantung": digantung,
        "Diterima": diterima
    }



# ==============================
# GUI
# ==============================

class FuzzyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem seleksi Beasiswa - Fuzzy Mamdani Realtime (Kurva)")
        self.resize(1200,800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Sliders
        self.s1 = self.slider("Input 1 (Skala Faktor)")
        self.s2 = self.slider("Input 2 (Prestasi)")
        self.s3 = self.slider("Input 3 (Konsistensi)")
        self.s4 = self.slider("Input 4 (Progres/Stabilitas)")

        # Output Label
        self.out = QLabel("Output: -")
        self.out.setStyleSheet("font-size: 22px; font-weight: bold; padding:10px;")
        self.layout.addWidget(self.out)

        # ---- Figure A (Membership Input + titik)
        self.figA = Figure(figsize=(5,4))
        self.canvasA = FigureCanvas(self.figA)
        self.axA = self.figA.add_subplot(111)
        self.layout.addWidget(self.canvasA)

        # ---- Figure C (Output Curves)
        self.figC = Figure(figsize=(5,4))
        self.canvasC = FigureCanvas(self.figC)
        self.axC = self.figC.add_subplot(111)
        self.layout.addWidget(self.canvasC)

        # === Label Status Akhir ===
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 26px; font-weight:bold; color: blue; margin:10px;")
        self.layout.addWidget(self.status_label)


        # Connect realtime update
        for s in (self.s1,self.s2,self.s3,self.s4):
            s.valueChanged.connect(self.update_plot)

        self.update_plot()


    def slider(self,name):
        box = QVBoxLayout()
        label = QLabel(name)
        s = QSlider(Qt.Horizontal)
        s.setRange(0,100)
        box.addWidget(label); box.addWidget(s)
        self.layout.addLayout(box)
        return s


    def update_plot(self):
        v1,v2,v3,v4 = self.s1.value(),self.s2.value(),self.s3.value(),self.s4.value()

        i1 = mf_input1(v1)
        i2 = mf_input2(v2)
        i3 = mf_input3(v3)
        i4 = mf_input4(v4)

        hasil = fuzzy_infer(i1,i2,i3,i4)
        keputusan = max(hasil, key=hasil.get)
        self.out.setText(f"Status: {keputusan} (μ={hasil[keputusan]:.2f})")
        # Update tulisan status di bawah grafik
        self.status_label.setText(f"STATUS AKHIR :  {keputusan.upper()}   (μ = {hasil[keputusan]:.2f})")



        # ---- Grafik A: Membership + Titik Posisi
        self.axA.clear()
        R,S,T = mf_input1(X)
        self.axA.plot(X,R,label="Rendah")
        self.axA.plot(X,S,label="Sedang")
        self.axA.plot(X,T,label="Tinggi")
        self.axA.scatter([v1],[i1[0]],color='blue')
        self.axA.scatter([v1],[i1[1]],color='orange')
        self.axA.scatter([v1],[i1[2]],color='green')
        self.axA.set_title("Mode A: Posisi Nilai pada Kurva Membership")
        self.axA.set_ylim(0,1.05)
        self.axA.legend()

        # ---- Grafik C: Output Curves (Stylized Curves)

        self.axC.clear()

        # Domain untuk kurva

        # Membership shapes for output fuzzy sets
        gagal_curve     = np.piecewise(X, [X<=40, (X>40)&(X<60), X>=60], [1, lambda x:(60-x)/20, 0])
        digantung_curve = np.piecewise(X, [X<=40, (X>40)&(X<70), X>=70], [0, lambda x:(x-40)/30, 1])
        diterima_curve  = np.piecewise(X, [X<=60, (X>60)&(X<80), X>=80], [0, lambda x:(x-60)/20, 1])

        # Plot curves
        self.axC.plot(X, gagal_curve,     color="red",   linewidth=2, label="Gagal")
        self.axC.plot(X, digantung_curve, color="orange",linewidth=2, label="Digantung")
        self.axC.plot(X, diterima_curve,  color="green", linewidth=2, label="Diterima")

        # Highlight membership result
        self.axC.fill_between(X, 0, hasil["Gagal"], where=(X<40), color="red", alpha=0.25)
        self.axC.fill_between(X, 0, hasil["Digantung"], where=(X>40)&(X<70), color="orange", alpha=0.25)
        self.axC.fill_between(X, 0, hasil["Diterima"], where=(X>70), color="green", alpha=0.25)

        self.axC.set_title("Kurva Keputusan Seleksi Beasiswa (Fuzzy Output)")
        self.axC.set_ylim(0,1.05)
        self.axC.set_xlim(0,100)
        self.axC.set_xlabel("Nilai Keputusan (0 - 100)")
        self.axC.set_ylabel("Derajat Keanggotaan (μ)")
        self.axC.legend()
        self.canvasC.draw()



# =============================
# START APP
# =============================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FuzzyGUI()
    w.show()
    sys.exit(app.exec_())
