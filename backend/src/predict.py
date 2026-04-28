import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models/crypto_xgb.pkl"))

def predict_signal(features_dict):
    X = pd.DataFrame([features_dict])

    prob = model.predict_proba(X)[0][1]

    # 🔥 Smart signal system
    if prob > 0.75:
        signal = "STRONG BUY"
    elif prob > 0.6:
        signal = "BUY"
    elif prob > 0.5:
        signal = "WEAK BUY"
    else:
        signal = "SELL"

    return {
        "signal": signal,
        "confidence": float(prob)
    }