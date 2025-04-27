from flask_cors import CORS
import tensorflow as tf
from keras.losses import MeanSquaredError
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


# Define the custom objects for loading the model
custom_objects = {"mse": MeanSquaredError()}

try:
    # Load the trained model
    model_path = "E:/Bunny/Capstone_2025/machine_learning/fra_det_main/trained_fraud_detector_autoencoder.h5"
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

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

if __name__ == "__main__":
    app.run(debug=True, port=7919)  # Run on port 7919

# sample input:
# {
#   "input": [[0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4]]
# }