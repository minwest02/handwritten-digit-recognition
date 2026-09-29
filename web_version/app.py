# Created: 2026-09-29 22:52:50

import sys
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

# Import shared model from parent directory
sys.path.insert(0, '..')
from model import predict_digit, load_model

app = Flask(__name__)

# Load model on startup
print("Loading MNIST model...")
model = load_model()
print("Model loaded successfully!")


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for digit prediction.

    Request body (JSON):
    {
        "image": "base64_encoded_grayscale_image",
        "width": 280,
        "height": 280
    }

    Response (JSON):
    {
        "digit": 7,
        "confidence": 0.985
    }
    """
    try:
        data = request.json

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        width = data.get('width', 280)
        height = data.get('height', 280)

        # Convert bytes to numpy array (grayscale)
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))

        # Resize to 28x28 (MNIST format)
        image_pil = Image.fromarray(image_array, mode='L')
        resized = image_pil.resize((28, 28), Image.Resampling.LANCZOS)
        resized_array = np.array(resized)

        # Invert colors: white background becomes black (MNIST format)
        # Canvas has white bg with black strokes
        # MNIST expects black bg with white strokes
        inverted_array = 255 - resized_array

        # Check if canvas is blank
        if np.max(inverted_array) < 50:
            return jsonify({'error': 'Please draw a digit first!'}), 400

        # Run prediction
        predicted_digit, confidence = predict_digit(inverted_array)

        return jsonify({
            'digit': int(predicted_digit),
            'confidence': float(confidence)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
