import numpy as np
import tensorflow as tf
# Keras is the high-level, user-friendly API for building and training neural networks in TensorFlow.
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# We'll use the classic MNIST dataset—a collection of 70,000 handwritten digits (28x28 pixel images).
from tensorflow.keras.datasets import mnist
# Matplotlib is the standard library for creating visualizations and plotting data in Python.
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. Parameters and Data Preparation: Getting the Data Ready ---
# ==============================================================================

# Define the size of our 'bottleneck' layer. 
# This is the small, compressed representation of the data, known as the **Latent Space**. 
# We are shrinking the 784-pixel image down to just 32 numbers!
LATENT_DIM = 32

# Load the MNIST dataset. 
# The dataset comes with 60,000 training images and 10,000 test images.
# Note: Autoencoders are trained using **unsupervised learning**, so we ignore the labels (the actual digit, represented by '_').
(x_train, _), (x_test, _) = mnist.load_data()

# --- Data Preprocessing Steps (Critical for Neural Networks) ---

# 1. Normalize the image data: Convert the pixel values from the range 0-255 to 0.0-1.0.
# This crucial step prevents large numbers from dominating the learning process and helps the network converge faster.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Flatten the images: Convert the 28x28 2D array into a 784-element 1D vector.
# We must do this because we are using simple Dense (fully-connected) layers, which only accept 1D inputs.
# The `np.prod` calculates 28 * 28 = 784.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # Final shape: (60000 images, 784 pixels)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))     # Final shape: (10000 images, 784 pixels)

INPUT_DIM = x_train.shape[1] # Stores the value 784

print(f"✅ Data Preparation Complete:")
print(f"   Original Image Dimension (Flattened): {INPUT_DIM} pixels")
print(f"   Target Latent Space Dimension (Compressed): {LATENT_DIM} features")
print(f"   Total training samples available: {len(x_train)}")


# ==============================================================================
# --- 2. Build the Autoencoder Model: Defining the Architecture ---
# ==============================================================================

# ** 2a. Encoder: The 'Compression' Machine **
# The Encoder's job is to intelligently compress the large input (784 dimensions) into the small Latent Vector (32 dimensions).
input_img = Input(shape=(INPUT_DIM,), name='Input_784_Pixels') 

# Start the compression with a wider layer. 'relu' is the standard activation function for hidden layers.
encoded = Dense(128, activation='relu', name='Encoder_Hidden_1')(input_img)
# Further compression.
encoded = Dense(64, activation='relu', name='Encoder_Hidden_2')(encoded)

# The Latent Space: The Bottleneck. This is the **efficient representation**!
# This 32-dimensional vector contains the most important features of the image.
latent_vector = Dense(LATENT_DIM, activation='relu', name='Latent_Vector_32_Features')(encoded) 

# ** 2b. Decoder: The 'Decompression/Reconstruction' Machine **
# The Decoder must take the small 32-dimensional latent vector and try to expand it back into the original 784-pixel image.
# It acts as the mirror image of the Encoder.

# Begin expanding from the latent vector.
decoded = Dense(64, activation='relu', name='Decoder_Hidden_1')(latent_vector)
# Further expansion.
decoded = Dense(128, activation='relu', name='Decoder_Hidden_2')(decoded)

# Output Layer: Must exactly match the input dimension (784 pixels).
# We use the 'sigmoid' activation function here because it squeezes the output values back into the 0-1 range, matching our normalized input data.
output_img = Dense(INPUT_DIM, activation='sigmoid', name='Output_Reconstruction_784_Pixels')(decoded) 

# ** 2c. Autoencoder Model (End-to-End) **
# This is the full model that performs the complete loop: Input -> Compress -> Decompress -> Output.
autoencoder = Model(input_img, output_img, name='Full_Autoencoder_Model')

# ** 2d. Separate Encoder Model (for representation extraction) **
# We need this separate model to easily access *only* the compressed latent vector (z) after training.
encoder_model = Model(input_img, latent_vector, name='Encoder_Extractor_Model')


# ==============================================================================
# --- 3. Compile and Train: Teaching the Model to Compress ---
# ==============================================================================

# Compile the model: Prepare it for training.
# Optimizer: 'adam' is an efficient algorithm that adjusts the network's weights during training.
# Loss Function: 'mean_squared_error' (MSE) measures the average squared difference between the Input and the Reconstructed Output.
# Our goal in training is to MINIMIZE this error.
autoencoder.compile(optimizer='adam', loss='mse')

print("\n--- Training Autoencoder: Compression Learning Begins ---")
# Start the training process! The model is learning the compressed representation.
history = autoencoder.fit(
    # In unsupervised learning, the Input (x_train) and the Target Output (x_train) are the same!
    x_train, x_train,
    epochs=10,        # The number of times the model sees the entire dataset.
    batch_size=256,   # The number of samples processed before the model updates its weights.
    shuffle=True,     # Shuffling the data prevents the model from learning the order of samples.
    validation_data=(x_test, x_test), # Check performance on unseen test data after each epoch.
    verbose=1         # Display detailed training progress.
)
print("✅ Training complete. The Autoencoder has successfully learned to find the efficient representation.")


# ==============================================================================
# --- 4. Demonstration and Visualization: Proving the Representation Works ---
# ==============================================================================

def visualize_reconstruction(original_data, encoded_model, full_model, n=10):
    """
    Demonstrates the power of the Autoencoder's latent representation.

    This function performs two main tasks:
    1. Extracts and prints the compressed **latent vector (z)** for a test image.
    2. Plots a side-by-side comparison of the original image and the reconstructed 
       image generated from that compressed vector, providing visual proof of success.

    Args:
        original_data (np.array): The preprocessed test data (x_test).
        encoded_model (tf.keras.Model): The trained Encoder model (for extracting the 32-dim vector).
        full_model (tf.keras.Model): The trained end-to-end Autoencoder model (for generating the reconstruction).
        n (int): The number of test samples to display in the plot.
    """
    
    # ** 4a. Extract Latent Representations (The core goal of the video!) **
    # We use the trained Encoder_Extractor to get the 32-dimensional codes for the test images.
    encoded_imgs = encoded_model.predict(original_data)

    # ** 4b. Reconstruct Images for Visual Proof **
    # We use the full Autoencoder to see how well it can turn the 32-dim code back into a 784-dim image.
    decoded_imgs = full_model.predict(original_data)

    # Print the shapes to highlight the massive dimensionality reduction
    print(f"\n📏 Dimensionality Analysis:")
    print(f"   Original test data shape: {original_data.shape} (784 dimensions)")
    print(f"   Encoded representation shape (Latent Vector): {encoded_imgs.shape} ({LATENT_DIM} dimensions)")
    print(f"   First 5 values of the compressed vector for the first image:\n   {encoded_imgs[0][:5]}...")
    print("   These 32 numbers are the model's efficient 'summary' of the image.")

    # --- Plotting the Results ---
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # --- 1. Display Original Image (The Input) ---
        ax = plt.subplot(2, n, i + 1)
        # Reshape the 784-vector back into a 28x28 image for display
        plt.imshow(original_data[i].reshape(28, 28))
        plt.title("Original")
        plt.gray() # Use grayscale
        ax.axis('off') # Turn off axis labels

        # --- 2. Display Reconstructed Image (The Output) ---
        ax = plt.subplot(2, n, i + 1 + n)
        # Reshape the reconstructed 784-vector back into a 28x28 image
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.title("Reconstructed")
        plt.gray()
        ax.axis('off')

    plt.suptitle(f"Visual Proof: Original vs. Reconstructed Images (Compressed to {LATENT_DIM} Dimensions)", fontsize=16, y=1.05)
    plt.show()

    print("\nConclusion: The visual similarity proves that the 32-dimensional latent vector is an effective and efficient data representation for the image!")


# Execute the visualization function using the test data and the trained models
visualize_reconstruction(x_test, encoder_model, autoencoder, n=10)