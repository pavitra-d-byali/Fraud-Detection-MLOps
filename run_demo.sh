#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/model/train.py --data-path data/creditcard.csv
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
