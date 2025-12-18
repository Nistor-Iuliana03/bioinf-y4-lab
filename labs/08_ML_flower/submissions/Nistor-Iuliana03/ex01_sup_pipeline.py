"""
Lab 08 — Supervised ML pipeline
Random Forest on gene expression

Student: Nistor-Iuliana03
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Config
# -----------------------
HANDLE = "Nistor-Iuliana03"

ROOT = Path(__file__).resolve().parents[4]
DATA_CSV = ROOT / f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv"

OUT_DIR = ROOT / f"labs/08_ML_flower/submissions/{HANDLE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(DATA_CSV)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y_enc = le.fit_transform(y)

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# -----------------------
# Train Random Forest
# -----------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# -----------------------
# Evaluation
# -----------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
)

print(report)
OUT_REPORT.write_text(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_CONFUSION, dpi=200)
plt.close()

# -----------------------
# Feature importance
# -----------------------
feat_imp = pd.DataFrame({
    "Gene": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

feat_imp.to_csv(OUT_FEATIMP, index=False)

print("Saved outputs in:", OUT_DIR)
