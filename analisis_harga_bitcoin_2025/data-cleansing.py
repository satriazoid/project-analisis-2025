import pandas as pd
import numpy as np
import re
from pathlib import Path

def detect_date_column(df):
    """
    Deteksi kolom tanggal yang memiliki pola dd/mm/yyyy
    """
    date_regex = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")

    for col in df.columns:
        if df[col].astype(str).str.contains(date_regex).any():
            return col
    return None

def detect_close_column(df):
    """
    Deteksi kolom harga penutupan (Close/Terakhir)
    """
    for name in ["Close", "Terakhir", "Price", "Last", "Penutupan"]:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    # fallback: pilih kolom angka terbesar
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    if len(numeric_cols):
        return numeric_cols[0]
    return None

def clean_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.replace(".", "")       # hapus pemisah ribuan 1.234.567
    s = s.replace(",", ".")      # ganti koma → titik untuk desimal
    s = s.replace("%", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def load_csv_safely(path):
    """
    Coba beberapa separator untuk memastikan CSV terbaca benar
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            # Jika jumlah kolom masuk akal, pakai ini
            if df.shape[1] >= 3:
                return df
        except:
            pass
    raise ValueError("CSV tidak bisa dibaca. Format separator tidak dikenali.")

def main():
    path = Path(r'C:\Code\Analisis\venv\analisis-harga-bitcoin-2025\dataset_clean.csv')
    if not path.exists():
        print("File dataset_btc.csv tidak ditemukan!")
        return

    df = load_csv_safely(path)

    # DETEKSI KOLOM TANGGAL
    col_date = detect_date_column(df)
    if col_date is None:
        raise ValueError("Tidak dapat menemukan kolom tanggal!")

    # DETEKSI KOLOM CLOSE
    col_close = detect_close_column(df)
    if col_close is None:
        raise ValueError("Tidak dapat menemukan kolom harga penutupan (Close/Terakhir)!")

    # RENAME
    df = df.rename(columns={col_date: "Date", col_close: "Close"})

    # KONVERSI TANGGAL FLEXIBLE
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # CLEAN NUMBERS
    df["Close"] = df["Close"].apply(clean_number)
    df = df.dropna(subset=["Date", "Close"])

    df = df.sort_values("Date").reset_index(drop=True)

    # MAKE LAG FEATURES
    for lag in range(1, 15):
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # ROLLING FEATURES
    for w in [7, 14]:
        df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["Close"].rolling(w).std().fillna(0)

    df = df.dropna().reset_index(drop=True)

    df.to_csv("dataset_btc_processed.csv", index=False)
    print("Berhasil! File tersimpan: dataset_btc_processed.csv")
    print(df.head())

if __name__ == "__main__":
    main()