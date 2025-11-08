# Real-Time Fraud Detection MLOps Pipeline 

**End-to-End Real-Time ML System** — A compact yet production-ready demo showcasing modern MLOps practices for fraud and anomaly detection.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Ready-success) ![Docker](https://img.shields.io/badge/Containerized-Yes-informational) ![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blueviolet)

---

## Highlights

* Automated training pipeline using `scikit-learn` with Random Forest and SMOTE (synthetic fallback).
* Real-time API inference served via FastAPI with request validation and type safety.
* Data-stream simulator that mimics live transactions (configurable rate and noise levels).
* MLOps ready: containerized with Docker and tested with GitHub Actions CI.
* Self-contained demo: if no dataset is provided, a synthetic one is generated automatically.

---

## Quick Start

```bash
git clone <repo-url> && cd repo
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train the Model

```bash
python src/model/train.py --data-path data/creditcard.csv
```

If the dataset is missing, a synthetic version will be created automatically.

### Launch FastAPI Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Simulate Real-Time Transactions

```bash
python src/simulator/streamer.py --url http://127.0.0.1:8000/predict --rate 2
```

### Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
-d '{"Time": 1, "Amount": 12.34, "V1": 0.1, "V2": -0.2, "V3": 0.05, "V4": -0.01, "V5": 0.02, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0}'
```

---

## Project Structure

```
├── src/
│   ├── model/       # Training and preprocessing
│   ├── api/         # FastAPI inference service
│   └── simulator/   # Streaming simulator
├── models/          # Saved model artifacts
├── data/            # Optional dataset
├── Dockerfile       # Containerization
└── .github/workflows/ci.yml  # CI/CD pipeline
```

---



## Future Enhancements

* Convert model to ONNX or TorchScript for low-latency inference.
* Add Prometheus metrics and Grafana dashboards for observability.
* Integrate MLflow or DVC for versioned model tracking.
* Deploy on Kubernetes or AWS Lambda for scalable serving.

---

## Author

**Pavitra Byali** — AI & ML Engineer | Passionate about building scalable, intelligent systems.



---

**License:** MIT
Minimal footprint • Real-time predictions • Production inspired
# Fraud-Detection-MLOps
