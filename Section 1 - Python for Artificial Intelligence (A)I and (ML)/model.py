# ----------------------------------------------------------------------
# Video 5: Building Your First AI Model - A Hands-on Project with TensorFlow
# Focus: The fundamental architecture (Sequential API) required for GANs.
# ----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. Load and Preprocess Data (The Foundation) ---
# We use the classic MNIST handwritten digits dataset for a simple classification task.
# This reinforces the data handling skills needed before moving to complex GAN data.

# Load the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("--- Data Loaded and Prepared ---")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# Normalize the image data (0-255) to a scale of 0 to 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# The data is 28x28 images. We need to flatten them into a vector of 784 features.
# This makes it suitable for our simple Dense (fully connected) network.
X_train = X_train.reshape((X_train.shape[0], 28 * 28))
X_test = X_test.reshape((X_test.shape[0], 28 * 28))

# --- 2. Build the Neural Network Model (Keras Sequential API) ---
# This structure is the direct precursor to building the Generator and Discriminator.
def build_simple_nn(input_dim):
    # The Sequential model stacks layers linearly.
    model = models.Sequential([
        # Input Layer and First Hidden Layer
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        
        # Second Hidden Layer (The 'Deep' part)
        layers.Dense(256, activation='relu'),
        
        # Output Layer: 10 classes (0 through 9) using Softmax for probabilities
        layers.Dense(10, activation='softmax')
    ])
    return model

# Initialize the model
input_dimension = 28 * 28
model = build_simple_nn(input_dimension)
print("-" * 50)
print("Neural Network Architecture Built:")
model.summary()
print("-" * 50)

# --- 3. Compile the Model ---
# Configure the learning process: optimizer, loss function, and metrics.
model.compile(
    optimizer='adam',
    # Use sparse_categorical_crossentropy for integer labels (0, 1, 2, etc.)
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model (The Learning Process) ---
print("Starting Model Training...")
history = model.fit(
    X_train, y_train,
    epochs=10,        # Number of full passes over the data
    batch_size=64,
    validation_data=(X_test, y_test), # Test performance on unseen data after each epoch
    verbose=1
)
print("Training Complete.")

# --- 5. Evaluate Final Performance ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc:.4f}")
print(f"Final Test Loss: {loss:.4f}")
