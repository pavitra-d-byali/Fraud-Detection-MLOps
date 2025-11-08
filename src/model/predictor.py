# helper utilities for prediction (not used by the API directly but useful for tests)
import joblib, pandas as pd
from pathlib import Path

def load_model(path=Path("models/fraud_detector.pkl")):
    data = joblib.load(path)
    return data['model'], data.get('scaler', None)

def predict_from_dict(d):
    model, scaler = load_model()
    df = pd.DataFrame([d])
    if scaler is not None:
        X = scaler.transform(df)
    else:
        X = df
    y = model.predict(X)
    prob = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
    return int(y[0]), (float(prob[0]) if prob is not None else None)
