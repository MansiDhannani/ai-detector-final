import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel

from feature_extractorv2 import FeatureExtractorV2

# -----------------------------------
# LOAD YOUR DATASET HERE
# -----------------------------------
# Example format:
# data = [{"code": "...", "label": 1}, ...]

data = joblib.load("training_data.pkl")  # <- change if needed

extractor = FeatureExtractorV2()

X = pd.DataFrame([extractor.extract(item["code"]) for item in data])
y = np.array([item["label"] for item in data])
feature_names = X.columns.tolist()

# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------
# Using stratify to maintain class balance in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
print("🔎 Starting XGBoost Hyperparameter Tuning...")
xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
print("Best XGBoost params:", grid_search.best_params_)

# RF Backup for comparison
rf_backup = RandomForestClassifier(n_estimators=100, random_state=42)
rf_backup.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_backup.predict(X_test))
print(f"RF backup accuracy: {rf_acc:.4f}")

# Calibrate probabilities
print("⚖️ Calibrating XGBoost probabilities...")
calibrated_xgb = CalibratedClassifierCV(best_xgb, method='isotonic', cv=5)
calibrated_xgb.fit(X_train, y_train)

# -----------------------------------
# TRAIN CODEBERT CLASSIFIER
# -----------------------------------
print("📥 Loading CodeBERT for feature extraction...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
codebert_model = AutoModel.from_pretrained('microsoft/codebert-base').to(device)

def get_embeddings(codes):
    codebert_model.eval()
    all_embeddings = []
    batch_size = 4
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = codebert_model(**encoded)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(all_embeddings, dim=0)

print("🧠 Extracting embeddings for CodeBERT classifier...")
raw_codes = [item["code"] for item in data]
train_indices = train_test_split(range(len(data)), test_size=0.2, random_state=42, stratify=y)[0]
train_codes = [raw_codes[i] for i in train_indices]
train_embeddings = get_embeddings(train_codes)

codebert_clf = LogisticRegression(max_iter=1000)
codebert_clf.fit(train_embeddings.numpy(), y_train)
print("✅ CodeBERT classifier trained")

# -----------------------------------
# EVALUATION
# -----------------------------------
y_pred = calibrated_xgb.predict(X_test)
y_prob = calibrated_xgb.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------------------
# SAVE MODEL
# -----------------------------------
joblib.dump({
    "xgb_base_model": best_xgb,
    "xgb_model": calibrated_xgb,
    "X_train": pd.DataFrame(X_train, columns=feature_names),
    "y_train": y_train,
    "codebert_clf": codebert_clf,
    "codebert_embeddings": train_embeddings,
    "codebert_y_train": y_train,
    "weight_xgb": 0.8,
    "weight_codebert": 0.2
}, "saved_models/hybrid_ai_detector_ensemble.pkl")
joblib.dump(feature_names, "saved_models/feature_names_v2.pkl")

print("\n✅ Feature model v2 saved successfully")

import joblib

MODEL_PATH = "hybrid_ai_detector.pkl"

joblib.dump(
    {
        "xgb_model": calibrated_xgb,
        "codebert_clf": codebert_clf,
        "weight_xgb": 0.5,
        "weight_codebert": 0.5
    },
    MODEL_PATH
)

print(f"✅ Models saved successfully to {MODEL_PATH}")
