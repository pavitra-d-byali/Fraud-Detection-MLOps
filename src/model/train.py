print("### RUNNING XGBOOST TRAINING FILE ###")

import argparse
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# Config — import shared constants from config module
# -------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import (
    BASE_DIR,
    COST_FN,
    COST_FP,
    LOGS_DIR,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    TEST_DATA_PATH,
    TEST_SIZE,
)

# -------------------------------------------------------------------
# Logging — absolute path so it works regardless of CWD
# -------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOGS_DIR / "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


# -------------------------------------------------------------------
# Cost-sensitive threshold evaluation
# -------------------------------------------------------------------
def expected_loss(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN


# -------------------------------------------------------------------
# Cross-validation
# -------------------------------------------------------------------
def cross_validate(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]

        roc_scores.append(roc_auc_score(y_val, y_prob))
        pr_scores.append(average_precision_score(y_val, y_prob))

        logging.info(f"Fold {fold+1}: ROC={roc_scores[-1]:.4f}, PR={pr_scores[-1]:.4f}")

    return np.mean(roc_scores), np.mean(pr_scores)


# -------------------------------------------------------------------
# Main training routine
# -------------------------------------------------------------------
def train(args):
    data_path = Path(args.data_path) if args.data_path else BASE_DIR / "data" / "creditcard.csv"

    mlflow.set_experiment("Fraud Detection XGBoost")

    with mlflow.start_run():

        X, y = load_data(data_path)

        # Cross-validation
        cv_roc, cv_pr = cross_validate(X, y)

        logging.info(f"CV ROC-AUC: {cv_roc:.4f}")
        logging.info(f"CV PR-AUC:  {cv_pr:.4f}")

        mlflow.log_metric("cv_roc_auc", cv_roc)
        mlflow.log_metric("cv_pr_auc", cv_pr)

        # Final train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        # Cost-optimal threshold selection
        thresholds = np.linspace(0.01, 0.5, 50)
        losses = [expected_loss(y_test, y_prob, t) for t in thresholds]

        best_idx = int(np.argmin(losses))
        best_threshold = float(thresholds[best_idx])

        y_final = (y_prob >= best_threshold).astype(int)

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)

        logging.info(f"Final ROC-AUC: {roc:.4f}")
        logging.info(f"Final PR-AUC:  {pr:.4f}")
        logging.info(f"Best threshold: {best_threshold:.4f}")

        mlflow.log_metric("final_roc_auc", roc)
        mlflow.log_metric("final_pr_auc", pr)
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_param("cost_fn", COST_FN)

        print("\n=== FINAL EVALUATION ===")
        print(f"Best Threshold: {best_threshold:.3f}")
        print(classification_report(y_test, y_final))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_final))
        print(f"ROC-AUC: {roc:.3f}")
        print(f"PR-AUC:  {pr:.3f}")

        # Save artifacts — absolute path so CWD doesn't matter
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "threshold": best_threshold,
                "cost_fp": COST_FP,
                "cost_fn": COST_FN,
                "features": list(X.columns),
            },
            MODEL_PATH,
        )

        joblib.dump(
            {"X_test": X_test, "y_test": y_test},
            TEST_DATA_PATH,
        )

        mlflow.xgboost.log_model(model, "model")

        print(f"Model saved → {MODEL_PATH}")
        print(f"Test split saved → {TEST_DATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to creditcard.csv (defaults to data/creditcard.csv in project root)",
    )
    args = parser.parse_args()
    train(args)
