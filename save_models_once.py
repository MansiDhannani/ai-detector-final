from hybrid_ai_detector import (
    calibrated_xgb,
    codebert_clf,
)
import joblib

joblib.dump(
    {
        "xgb_model": calibrated_xgb,
        "codebert_clf": codebert_clf,
        "weight_xgb": 0.8,
        "weight_codebert": 0.2,
    },
    "hybrid_ai_detector.pkl"
)

print("✅ hybrid_ai_detector.pkl saved successfully")
