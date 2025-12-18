"""
Lab 08 — Unsupervised Learning
KMeans clustering + crosstab vs labels

Student: Nistor-Iuliana03
"""

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Config
# -----------------------
HANDLE = "Nistor-Iuliana03"

ROOT = Path(__file__).resolve().parents[4]
DATA_CSV = ROOT / f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv"

OUT_DIR = ROOT / f"labs/08_ML_flower/submissions/{HANDLE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(DATA_CSV)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y_enc = le.fit_transform(y)

# -----------------------
# KMeans
# -----------------------
n_clusters = len(le.classes_)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init="auto"
)

clusters = kmeans.fit_predict(X.values)

# -----------------------
# Crosstab
# -----------------------
df_res = pd.DataFrame({
    "Label": y,
    "Cluster": clusters
})

ctab = pd.crosstab(df_res["Label"], df_res["Cluster"])
print(ctab)

ctab.to_csv(OUT_CROSSTAB)
print("Saved crosstab to:", OUT_CROSSTAB)
