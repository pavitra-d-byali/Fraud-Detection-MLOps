from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# -----------------------------
# Base paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "fraud_detector.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# -----------------------------
# Load trained artifacts
# -----------------------------
if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run `python src/model/train.py` first.")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
THRESHOLD = artifact["threshold"]
COST_FP = artifact["cost_fp"]
COST_FN = artifact["cost_fn"]
FEATURES = artifact["features"]

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Fraud Detection API")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# -----------------------------
# Request schema
# -----------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# -----------------------------
# Frontend route
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": THRESHOLD,
        "cost_policy": {
            "cost_fp": COST_FP,
            "cost_fn": COST_FN
        }
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(tx: Transaction):
    try:
        x = np.array([getattr(tx, f) for f in FEATURES], dtype=float).reshape(1, -1)
        prob = float(model.predict_proba(x)[0][1])
        decision = int(prob >= THRESHOLD)

        return {
            "fraud_probability": round(prob, 4),
            "threshold": THRESHOLD,
            "decision": "FRAUD" if decision == 1 else "LEGIT",
            "cost_policy": {
                "cost_fp": COST_FP,
                "cost_fn": COST_FN
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))