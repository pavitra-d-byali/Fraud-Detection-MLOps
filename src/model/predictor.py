# Helper utilities for prediction (used by tests and ad-hoc scripts)
import sys
from pathlib import Path

import joblib
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import MODEL_PATH


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Load the full saved artifact dict (model + metadata)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run `python src/model/train.py` first."
        )
    return joblib.load(path)


def predict_from_dict(d: dict) -> tuple[int, float | None]:
    """
    Given a dict of feature_name → value, return (class_label, fraud_probability).
    Feature order is enforced from the saved artifact so column mismatch is impossible.
    """
    artifact = load_artifact()
    model = artifact["model"]
    features = artifact["features"]      # ordered list from training
    threshold = artifact["threshold"]

    df = pd.DataFrame([d])[features]     # enforce training column order

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[:, 1][0])
        label = int(prob >= threshold)
    else:
        label = int(model.predict(df)[0])

    return label, prob
