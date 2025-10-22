import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# 1. Prepare Data (Using MNIST as a proxy for simple image data)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Flatten the 28x28 images to 784-dimensional vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the size of the latent space (e.g., 32 dimensions)
encoding_dim = 32
input_dim = x_train.shape[1]

# 2. Build the Autoencoder Architecture

# --- Encoder ---
input_layer = Input(shape=(input_dim,))
# Encodes the input to the latent representation
encoded = Dense(128, activation='relu')(input_layer)
latent_vector = Dense(encoding_dim, activation='relu', name='latent_space')(encoded)

# --- Decoder ---
# Decodes the latent representation back to the original dimension
decoded = Dense(128, activation='relu')(latent_vector)
output_layer = Dense(input_dim, activation='sigmoid')(decoded) # Sigmoid for pixel values [0, 1]

# --- Autoencoder Model ---
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Optional: Separate Encoder and Decoder models for later use
encoder_model = Model(inputs=input_layer, outputs=latent_vector)

# 3. Train the Model
print("Starting Autoencoder Training...")
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 4. Demonstrate Reconstruction (Optional Visualization)
reconstructed_images = autoencoder.predict(x_test[:10])

# Show original and reconstructed images side-by-side
print("\nExample Reconstructions (Original vs. Reconstructed):")
for i in range(10):
    original = x_test[i].reshape(28, 28)
    reconstructed = reconstructed_images[i].reshape(28, 28)

    plt.figure(figsize=(2, 1))
    
    ax = plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("Original")

    ax = plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("Reconstructed")
    plt.show()