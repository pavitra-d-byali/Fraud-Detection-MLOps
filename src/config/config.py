from pathlib import Path

# Project root is two levels up from this file (src/config/config.py)
BASE_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_detector.pkl"
TEST_DATA_PATH = BASE_DIR / "models" / "test_data.pkl"
LOGS_DIR = BASE_DIR / "logs"

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5
MODEL_NAME = "xgboost"

# Cost-sensitive threshold tuning
COST_FP = 1    # cost of a false positive (blocking a legit transaction)
COST_FN = 50   # cost of a false negative (missing a fraud)
