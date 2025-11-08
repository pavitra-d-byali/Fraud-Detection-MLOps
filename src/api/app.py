from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, pandas as pd
from pathlib import Path

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = Path("models/fraud_detector.pkl")

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

model = None
scaler = None
try:
    data = joblib.load(MODEL_PATH)
    model = data['model']
    scaler = data.get('scaler', None)
except Exception:
    model = None

@app.get("/")
def read_root():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run `python src/model/train.py` to create models/fraud_detector.pkl")
    df = pd.DataFrame([tx.dict()])
    features = df.drop(columns=[])
    if scaler is not None:
        features = scaler.transform(features)
    y_pred = model.predict(features)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(features)[:,1].tolist()
    return {"fraudulent": bool(int(y_pred[0])), "probability": y_prob[0] if y_prob is not None else None}
