import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dense, Conv2D, Conv2DTranspose, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential

# Configuration constants
IMAGE_SHAPE = (64, 64, 3) # Example: 64x64 RGB images for phishing templates
LATENT_DIM = 100 # Size of the random noise vector (z)

# --- 1. The Generator Model (Upsampling) ---
def build_generator(latent_dim, image_shape):
    model = Sequential([
        # Start with a dense layer to project and reshape the noise
        Dense(4 * 4 * 256, input_shape=(latent_dim,)),
        LeakyReLU(),
        BatchNormalization(),
        Reshape((4, 4, 256)), # Start with 4x4 image with 256 filters

        # Upsampling block 1: 4x4 -> 8x8
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(),
        BatchNormalization(),

        # Upsampling block 2: 8x8 -> 16x16
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(),
        BatchNormalization(),
        
        # Upsampling block 3: 16x16 -> 32x32
        Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(),
        BatchNormalization(),

        # Upsampling block 4: 32x32 -> 64x64 (Final Output)
        Conv2DTranspose(image_shape[-1], (4, 4), strides=(2, 2), padding='same', activation='tanh') 
        # Tanh maps output pixel values to [-1, 1]
    ], name='generator')
    
    return model

# --- 2. The Discriminator Model (Downsampling) ---
def build_discriminator(image_shape):
    model = Sequential([
        # Downsampling block 1: 64x64 -> 32x32
        Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=image_shape),
        LeakyReLU(alpha=0.2), # Use Leaky ReLU in Discriminator

        # Downsampling block 2: 32x32 -> 16x16
        Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),

        # Downsampling block 3: 16x16 -> 8x8
        Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        
        # Downsampling block 4: 8x8 -> 4x4
        Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),

        # Output: Flatten and Dense layer for binary classification
        Flatten(),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification (Real/Fake)
    ], name='discriminator')
    
    return model

# Instantiate the models (for demonstration)
generator = build_generator(LATENT_DIM, IMAGE_SHAPE)
discriminator = build_discriminator(IMAGE_SHAPE)

generator.summary()
discriminator.summary()