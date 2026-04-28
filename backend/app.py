from flask import Flask, request, jsonify
from flask_cors import CORS
from src.predict import predict_signal
from src.data_fetcher import get_latest_features

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        coin = data.get("coin", "bitcoin").lower()

        # 🔥 REAL DATA FEATURES
        features = get_latest_features(coin)

        result = predict_signal(features)

        return jsonify({
            "coin": coin,
            "features": features,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)