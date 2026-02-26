import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load saved bundle
bundle = joblib.load("models/fraud_detector.pkl")

model = bundle["model"]
threshold = bundle["threshold"]
feature_list = bundle["features"]

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Ensure feature order matches training
X = df[feature_list]
y = df["Class"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Apply custom threshold
y_pred = (y_prob >= threshold).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))