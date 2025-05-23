from flask_cors import CORS
import tensorflow as tf
from keras.losses import MeanSquaredError
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Define the custom objects for loading the model
custom_objects = {"mse": MeanSquaredError()}

try:
    # Load the trained model using a relative path
    model_path = os.path.join(os.path.dirname(__file__), "trained_fraud_detector_autoencoder.h5")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return jsonify({'message': 'API is up and running'})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Invalid input data"}), 400

        # Convert input data into a TensorFlow-compatible format
        input_data = tf.convert_to_tensor(data["input"])

        # Get predictions from the model
        predictions = model.predict(input_data)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# No app.run() here.
# Waitress will serve this app separately via another script.
