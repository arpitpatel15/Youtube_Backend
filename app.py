from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from detector_api import predict_video_from_url, MODEL_PATH

app = Flask(__name__)

# ------------------------------
#  SAFE JSON CONVERSION FUNCTION
# ------------------------------
def to_native(obj):
    """Convert numpy types to Python native types for JSON."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def sanitize_result(result):
    """Handle dict, list, or single value outputs from model."""
    if isinstance(result, dict):
        return {k: to_native(v) for k, v in result.items()}
    if isinstance(result, list):
        return [to_native(v) for v in result]
    return to_native(result)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("🚀 Model Loaded Successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🤖 YouTube Fake Video Detector API is running!",
        "usage": {
            "POST /predict": {
                "url": "YouTube video link"
            }
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "url" not in data:
            return jsonify({"error": "Missing key: 'url'"}), 400

        youtube_url = data["url"]

        # Run prediction
        result = predict_video_from_url(youtube_url)

        # Convert numpy → python types
        cleaned_result = sanitize_result(result)

        return jsonify(cleaned_result), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Prediction failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
