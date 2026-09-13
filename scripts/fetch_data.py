"""Fetch UCI Heart Disease data (4 collection sites) into data/<site>.csv.
Public, anonymized research data: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
import io, urllib.request
from pathlib import Path
import pandas as pd

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]

def main():
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    for site, fname in SITES.items():
        raw = urllib.request.urlopen(BASE + fname, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw), header=None, names=COLS, na_values="?")
        df.to_csv(out / f"{site}.csv", index=False)
        print(f"{site}: {len(df)} rows -> {out / (site + '.csv')}")

if __name__ == "__main__":
    main()
