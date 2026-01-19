print("### RUNNING XGBOOST TRAINING FILE ###")

import argparse
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

COST_FP = 1
COST_FN = 50


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def expected_loss(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN


def train(args):
    X, y = load_data(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.01, 0.5, 50)
    losses = [expected_loss(y_test, y_prob, t) for t in thresholds]

    best_idx = int(np.argmin(losses))
    best_threshold = float(thresholds[best_idx])

    y_final = (y_prob >= best_threshold).astype(int)

    print("\n=== Final Evaluation (XGBoost + Cost) ===")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Expected Loss: {losses[best_idx]}")
    print(classification_report(y_test, y_final, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_final))

    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_prob):.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "cost_fp": COST_FP,
            "cost_fn": COST_FN,
            "features": list(X.columns)
        },
        "models/fraud_detector.pkl"
    )

    print("Saved model to models/fraud_detector.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    args = parser.parse_args()
    train(args)
