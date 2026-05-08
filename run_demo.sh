#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_demo.sh — Quick local demo (requires creditcard.csv in data/)
# ──────────────────────────────────────────────────────────────
set -e

echo "==> Setting up virtual environment..."
python -m venv .venv

# Activate (Linux/macOS). On Windows use: .venv\Scripts\activate
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Training model..."
python src/model/train.py --data-path data/creditcard.csv

echo "==> Starting FastAPI backend (port 8000)..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Give backend a moment to start
sleep 3

echo "==> Starting Streamlit frontend (port 8501)..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "    Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ Demo running!"
echo "   API:       http://localhost:8000"
echo "   Swagger:   http://localhost:8000/docs"
echo "   Frontend:  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
