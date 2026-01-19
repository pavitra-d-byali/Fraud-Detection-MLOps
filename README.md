# 🚀 Real-Time Fraud Detection MLOps Pipeline

## 📌 Project Overview

This project implements a **real-world fraud detection pipeline** where model decisions are driven by **business cost**, not just accuracy.

Unlike typical ML demos, this system explicitly optimizes **expected financial loss**, making it suitable for real banking and payment scenarios.

**This is not a notebook demo — it is a deployable ML system.**

---

## 🎯 Key Focus Areas

- Severe class imbalance handling  
- Cost-sensitive learning (False Negatives ≫ False Positives)  
- Threshold optimization based on expected loss  
- Training–serving consistency  
- Deployable real-time inference API  

---

## ✨ Key Features

### 🔹 Cost-Sensitive Training
- XGBoost with `scale_pos_weight`
- Explicit business cost modeling  
  - `COST_FN = 50`
  - `COST_FP = 1`
- Decision threshold optimized using **expected loss**, not accuracy

### 🔹 Production-Grade Inference
- FastAPI service enforcing the **trained threshold**
- No training–serving skew
- Deterministic, auditable predictions

### 🔹 MLOps Foundations
- Versioned model artifacts (model + threshold + cost policy)
- Dockerized deployment
- CI pipeline via GitHub Actions

### 🔹 Real-Time Simulation
- Transaction streaming simulator for live inference testing

---

## 🧠 Why Cost-Sensitive Fraud Detection?

In fraud detection systems:

- Missing fraud (**False Negative**) is **far more expensive**
- Flagging a legitimate transaction (**False Positive**) is comparatively cheap
- Optimizing **accuracy or F1-score alone leads to bad business decisions**

This system explicitly minimizes:

```text
Expected Loss = (False Positives × COST_FP) + (False Negatives × COST_FN)
## 🗂️ Project Structure

```text
├── src/
│   ├── model/       # XGBoost training + cost-based threshold tuning
│   ├── api/         # FastAPI inference service
│   └── simulator/   # Real-time transaction streamer
├── models/          # Trained model artifacts
├── data/            # Credit card fraud dataset
├── Dockerfile       # Containerization
└── .github/workflows/ci.yml  # CI pipeline
## ⚡ Quick Start

```bash
git clone https://github.com/pavitra-d-byali/Fraud-Detection-MLOps.git
cd Fraud-Detection-MLOps

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
## 🧪 Train the Model

```bash
python src/model/train.py --data-path data/creditcard.csv
### 📊 Training Output Includes

- Optimized decision threshold  
- Confusion matrix  
- ROC-AUC and PR-AUC  
- Expected loss under business costs  

### 📦 Saved Artifact

```text
models/fraud_detector.pkl
## 🚦 Run the Inference API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
📘 API Documentation
Access the interactive Swagger UI at:

arduino
Copy code
http://127.0.0.1:8000/docs
🔁 Simulate Real-Time Transactions
bash
Copy code
python src/simulator/streamer.py \
  --url http://127.0.0.1:8000/predict \
  --rate 2
📡 Example API Request
bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "Time": 1.0,
  "Amount": 149.62,
  "V1": -1.35,
  "V2": 0.12,
  "V3": 0.45,
  "V4": -0.23,
  "V5": 0.56,
  "V6": -0.11,
  "V7": 0.32,
  "V8": 0.14,
  "V9": -0.88,
  "V10": -1.02,
  "V11": 0.45,
  "V12": -0.67,
  "V13": 0.23,
  "V14": -1.55,
  "V15": 0.31,
  "V16": -0.44,
  "V17": -1.23,
  "V18": 0.12,
  "V19": -0.56,
  "V20": 0.09,
  "V21": -0.12,
  "V22": 0.33,
  "V23": -0.21,
  "V24": 0.45,
  "V25": -0.12,
  "V26": 0.08,
  "V27": -0.04,
  "V28": 0.02
}'
📈 Model Characteristics (Honest)
High recall for fraud detection

Lower precision by design

Threshold chosen to minimize business loss

Behavior aligned with real banking fraud systems

✔️ This trade-off is intentional and correct

🔮 Future Enhancements
SHAP-based model explainability

MLflow experiment tracking

Data drift detection

API authentication & rate limiting

Kubernetes deployment

Prometheus + Grafana monitoring

👤 Author
Pavitra Byali
AI & ML Engineer
Focused on building production-grade, business-aware ML systems

📄 License
MIT License