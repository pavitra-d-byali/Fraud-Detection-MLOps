import argparse, time, json, requests, random
import numpy as np

SAMPLE_TEMPLATE = {
    "Time": 1.0,
    "Amount": 10.0,
}
for i in range(1,29):
    SAMPLE_TEMPLATE[f"V{i}"] = 0.0

def random_tx():
    tx = SAMPLE_TEMPLATE.copy()
    tx["Time"] = random.uniform(0, 172792)
    tx["Amount"] = float(abs(random.gauss(50, 80)))
    for i in range(1,29):
        tx[f"V{i}"] = float(random.gauss(0,1))
    return tx

def stream(url, rate=1.0):
    while True:
        tx = random_tx()
        try:
            resp = requests.post(url, json=tx, timeout=5)
            print("sent, status:", resp.status_code, resp.text)
        except Exception as e:
            print("failed to send:", e)
        time.sleep(1.0/max(1, rate))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--rate", type=float, default=1.0, help="transactions per second")
    args = parser.parse_args()
    stream(args.url, args.rate)
