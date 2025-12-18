"""
Lab 08 — Logistic Regression vs Random Forest
Comparatie modele pe expresie genica

Student: Nistor-Iuliana03
"""

from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# -----------------------
# Config
# -----------------------
HANDLE = "Nistor-Iuliana03"

ROOT = Path(__file__).resolve().parents[4]
DATA_CSV = ROOT / f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv"

OUT_DIR = ROOT / f"labs/08_ML_flower/submissions/{HANDLE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"

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
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# -----------------------
# Random Forest
# -----------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -----------------------
# Logistic Regression
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    n_jobs=-1
)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)

# -----------------------
# Reports
# -----------------------
report_rf = classification_report(
    y_test,
    y_pred_rf,
    target_names=le.classes_
)

report_lr = classification_report(
    y_test,
    y_pred_lr,
    target_names=le.classes_
)

combined = (
    "=== Random Forest ===\n"
    + report_rf
    + "\n\n=== Logistic Regression ===\n"
    + report_lr
)

print(combined)
OUT_REPORT.write_text(combined)

print("Saved comparison report to:", OUT_REPORT)
