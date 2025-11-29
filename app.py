from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os
import numpy as np
from detector_api import predict_video_from_url, MODEL_PATH

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://www.youtube.com",
    "chrome-extension://*"
]}})
# ------------------------------
#  SAFE JSON CONVERSION FUNCTIONS
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

def sanitize_result(obj):
    """Recursively convert numpy types to native Python types."""
    # If it's a dict, verify every value recursively
    if isinstance(obj, dict):
        return {k: sanitize_result(v) for k, v in obj.items()}
    
    # If it's a list, verify every element recursively
    if isinstance(obj, list):
        return [sanitize_result(v) for v in obj]
    
    # If it's a single value, use to_native
    return to_native(obj)

if not os.path.exists(MODEL_PATH):
    # Depending on deployment, you might want to print a warning rather than crash
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)
    print("🚀 Model Loaded Successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "YouTube Fake Video Detector API is running!",
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

        # Convert numpy -> python types (Recursive Fix)
        cleaned_result = sanitize_result(result)

        return jsonify(cleaned_result), 200

    except Exception as e:
        # Print error to logs for debugging on Render
        print(f"Server Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "Prediction failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)