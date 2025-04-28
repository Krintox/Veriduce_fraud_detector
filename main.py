import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Disable GPU (optional to avoid unnecessary errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://veri-duce.vercel.app"}})

# Load model
MODEL_PATH = "trained_fraud_detector_autoencoder.h5"
model = None

try:
    if os.path.exists(MODEL_PATH):
        # Important: compile=False while loading
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define route
@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)  # Reshape if needed
        reconstruction = model.predict(input_data)
        loss = np.mean(np.power(input_data - reconstruction, 2), axis=1)

        # Threshold for fraud detection (adjust as needed)
        threshold = 0.01
        prediction = "Fraud" if loss > threshold else "Not Fraud"

        return jsonify({
            "loss": float(loss),
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port)  # NO debug=True
