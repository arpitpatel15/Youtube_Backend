from flask import Flask, request, jsonify
import joblib
import os
from detector_api import (
    predict_video_from_url,MODEL_PATH
)
app = Flask(__name__)
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
        result = predict_video_from_url(youtube_url)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
