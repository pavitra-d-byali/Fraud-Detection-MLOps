print("### RUNNING XGBOOST TRAINING FILE ###")

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier

# ------------------ CONFIG ------------------
COST_FP = 1
COST_FN = 50
N_SPLITS = 5
# --------------------------------------------


# ------------------ LOGGING ------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# ---------------------------------------------


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def expected_loss(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN


def cross_validate(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    roc_scores = []
    pr_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

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

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]

        roc_scores.append(roc_auc_score(y_val, y_prob))
        pr_scores.append(average_precision_score(y_val, y_prob))

    return np.mean(roc_scores), np.mean(pr_scores)


def train(args):
    mlflow.set_experiment("Fraud Detection XGBoost")

    with mlflow.start_run():

        X, y = load_data(args.data_path)

        # Cross-validation
        cv_roc, cv_pr = cross_validate(X, y)

        logging.info(f"CV ROC-AUC: {cv_roc}")
        logging.info(f"CV PR-AUC: {cv_pr}")

        mlflow.log_metric("cv_roc_auc", cv_roc)
        mlflow.log_metric("cv_pr_auc", cv_pr)

        # Final Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)

        logging.info(f"Final ROC-AUC: {roc}")
        logging.info(f"Final PR-AUC: {pr}")

        mlflow.log_metric("final_roc_auc", roc)
        mlflow.log_metric("final_pr_auc", pr)
        mlflow.log_param("best_threshold", best_threshold)

        print("\n=== FINAL EVALUATION ===")
        print(f"Best Threshold: {best_threshold:.3f}")
        print(classification_report(y_test, y_final))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_final))
        print(f"ROC-AUC: {roc:.3f}")
        print(f"PR-AUC: {pr:.3f}")

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

        # Save test split for consistent evaluation
        joblib.dump(
            {"X_test": X_test, "y_test": y_test},
            "models/test_data.pkl"
        )

        mlflow.xgboost.log_model(model, "model")

        print("Model and test split saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    args = parser.parse_args()
    train(args)
# TODO: integrate config module
