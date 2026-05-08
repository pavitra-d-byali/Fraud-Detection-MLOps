"""
Standalone evaluation script.
Run from any directory:  python src/evaluation.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE

# ------------------------------------------------------------------
# Load saved artifact
# ------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run `python src/model/train.py` first."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
feature_list = bundle["features"]
cost_fp = bundle["cost_fp"]
cost_fn = bundle["cost_fn"]

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Download creditcard.csv from Kaggle and place it in data/."
    )

df = pd.read_csv(DATA_PATH)
X = df[feature_list]   # enforce training column order
y = df["Class"]

# ------------------------------------------------------------------
# Reproduce the same stratified split used during training
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# Predict with the cost-optimal threshold
# ------------------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("=" * 60)
print(f"Model artifact : {MODEL_PATH}")
print(f"Best threshold : {threshold:.4f}")
print(f"Cost policy    : FP={cost_fp}, FN={cost_fn}")
print("=" * 60)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")