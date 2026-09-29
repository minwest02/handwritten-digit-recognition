import os
import tensorflow as tf
import numpy as np

# Load MNIST dataset and build model
def build_model():
    """
    Build a simple CNN model for digit classification.
    Input: 28x28 grayscale images
    Output: 10 classes (digits 0-9)
    """
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 output classes (0-9)
    ])

    return model


def train_model(epochs=5):
    """
    Load MNIST dataset, build model, and train for specified epochs.
    Saves the trained model to mnist_model.h5.
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape to (num_samples, 28, 28, 1) for Conv2D input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Training MNIST model (this may take a few minutes)...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1)

    # Save trained model
    try:
        model.save('mnist_model.h5')
        print("Model trained and saved to mnist_model.h5")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

    return model


def load_model():
    """
    Load pre-trained model from mnist_model.h5.
    If file doesn't exist, train a new model first.
    If file is corrupted, catch the exception and train a new model.
    """
    if os.path.exists('mnist_model.h5'):
        try:
            print("Loading pre-trained model from mnist_model.h5...")
            model = tf.keras.models.load_model('mnist_model.h5')
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Training new model...")
            return train_model(epochs=5)
    else:
        print("Model file not found. Training new model...")
        return train_model(epochs=5)


def predict_digit(image_array):
    """
    Predict digit from a 28x28 grayscale image array.

    Args:
        image_array: NumPy array of shape (28, 28) with values 0-255

    Returns:
        Tuple of (predicted_digit, confidence_score)
        - predicted_digit: int (0-9)
        - confidence_score: float (0.0-1.0)
    """
    # Load model (will use cached model if already loaded)
    model = load_model()

    # Normalize pixel values to [0, 1]
    normalized = image_array.astype('float32') / 255.0

    # Reshape to (1, 28, 28, 1) for model input
    input_tensor = normalized.reshape(1, 28, 28, 1)

    # Get predictions
    predictions = model.predict(input_tensor, verbose=0)

    # Extract predicted digit and confidence
    predicted_digit = int(np.argmax(predictions[0]))
    confidence_score = float(np.max(predictions[0]))

    return predicted_digit, confidence_score
