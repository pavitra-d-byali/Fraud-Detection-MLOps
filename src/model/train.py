import argparse, os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

def make_synthetic(n=1000, fraud_ratio=0.02, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(n, 30))
    # first column = Time, second = Amount, next 28 = V1..V28
    X[:,0] = rng.uniform(0, 172792, size=n)   # Time
    X[:,1] = rng.exponential(scale=100.0, size=n)  # Amount
    y = (rng.rand(n) < fraud_ratio).astype(int)
    cols = ['Time','Amount'] + [f'V{i}' for i in range(1,29)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name='Class')

def load_data(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if 'Class' not in df.columns:
            raise ValueError("CSV must contain 'Class' column")
        X = df.drop(columns=['Class'])
        y = df['Class']
        return X, y
    else:
        print("Data path not found: generating synthetic dataset for demo.")
        return make_synthetic(n=2000)

def main(args):
    X, y = load_data(args.data_path)
    # split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # handle imbalance with class_weight
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    # eval
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:,1] if hasattr(clf, "predict_proba") else None
    print(classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print("AUC-ROC:", auc)
    except Exception:
        pass

    # save
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, 'models/fraud_detector.pkl')
    print("Saved model to models/fraud_detector.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    args = parser.parse_args()
    main(args)
