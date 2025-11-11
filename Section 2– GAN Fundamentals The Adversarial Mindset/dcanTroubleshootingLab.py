import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy

# --- 1. Setup & Hyperparameters ---
LATENT_DIM = 100
IMG_SHAPE = (64, 64, 3)
BATCH_SIZE = 128
# >>> TROUBLESHOOTING PARAMETER: Label Smoothing Factor <<<
REAL_LABEL_SMOOTH = 0.9 

# Placeholder: Assume X_train is pre-processed to [-1, 1]
X_train = np.random.uniform(-1, 1, size=(5000, 64, 64, 3)).astype(np.float32) 

# --- 2. Simplified Placeholder Model Definitions (as before) ---

def make_generator_model():
    """Simple DCGAN Generator placeholder."""
    # ... (Model definition remains the same)
    noise_input = Input(shape=(LATENT_DIM,))
    x = Dense(4 * 4 * 512, use_bias=False)(noise_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((4, 4, 512))(x)
    x = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False)(x)
    return Model(noise_input, x)

def make_discriminator_model():
    """Simple DCGAN Discriminator placeholder."""
    # ... (Model definition remains the same)
    img_input = Input(shape=IMG_SHAPE)
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(img_input)
    x = LeakyReLU()(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(img_input, x)

generator = make_generator_model()
discriminator = make_discriminator_model()

# Define Optimizers and Losses (as before)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()
discriminator.compile(optimizer=d_optimizer, loss=loss_fn, metrics=['accuracy']) 

# Build the Adversarial Model (GAN) (as before)
z_input = Input(shape=(LATENT_DIM,))
g_output = generator(z_input)
d_output = discriminator(g_output)
gan = Model(z_input, d_output, name='gan')
discriminator.trainable = False 
gan.compile(optimizer=g_optimizer, loss=loss_fn) 


# --- 3. The Troubleshooting Training Loop (Modified with Label Smoothing) ---
@tf.function
def train_step_smoothed(images):
    # 1. Train Discriminator (Real samples) - APPLY LABEL SMOOTHING HERE
    # Instead of tf.ones((BATCH_SIZE, 1)), we use the smoothed label
    
    # >>> TROUBLESHOOTING CODE: Real labels are set to REAL_LABEL_SMOOTH (e.g., 0.9) <<<
    real_labels = tf.random.uniform(
        (BATCH_SIZE, 1), 
        minval=REAL_LABEL_SMOOTH, 
        maxval=1.0, 
        dtype=tf.float32
    )
    
    with tf.GradientTape() as tape:
        d_loss_real = loss_fn(real_labels, discriminator(images, training=True))
    d_gradients_real = tape.gradient(d_loss_real, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients_real, discriminator.trainable_variables))

    # 2. Train Discriminator (Fake samples) - FAKE LABELS REMAIN 0.0
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    fake_images = generator(noise, training=False)
    fake_labels = tf.zeros((BATCH_SIZE, 1))
    with tf.GradientTape() as tape:
        d_loss_fake = loss_fn(fake_labels, discriminator(fake_images, training=True))
    d_gradients_fake = tape.gradient(d_loss_fake, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients_fake, discriminator.trainable_variables))

    d_loss = d_loss_real + d_loss_fake

    # 3. Train Generator (The GAN Model) - G WANTS D TO OUTPUT 'REAL' (1)
    misleading_labels = tf.ones((BATCH_SIZE, 1)) 
    with tf.GradientTape() as tape:
        # Note: We do NOT smooth the labels for the Generator's loss.
        # G must still aim for a perfect '1.0' to ensure its images are highly realistic.
        g_loss = gan(noise, training=True)
    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, g_loss

# --- 4. Lab Demonstration & Conceptual Logging ---

def log_loss(d_loss_history, g_loss_history):
    """
    Conceptual function to visualize and check for convergence issues.
    In a real lab, you would use Matplotlib or TensorBoard to plot these.
    """
    print("\n--- Conceptual Loss Monitoring (Key to Troubleshooting) ---")
    
    # Check for D-Win (D loss near zero, G loss high)
    if d_loss_history[-1] < 0.1 and g_loss_history[-1] > 3.0:
        print("ALERT: Discriminator is winning! D loss is too low, G loss is exploding. Consider lowering D's learning rate.")
    
    # Check for G-Win (Both losses low, but images look bad - potential early Mode Collapse)
    elif d_loss_history[-1] < 0.5 and g_loss_history[-1] < 0.7:
        print("WARNING: Losses converging too fast. If images look identical, Mode Collapse may be occurring.")
    
    else:
        print("Losses appear to be balanced and training is progressing.")


# Example Training Simulation (Run for a single step to show the smoothed label)
print(f"Starting training simulation with REAL_LABEL_SMOOTH = {REAL_LABEL_SMOOTH}")

# Create an image batch from the dummy data
image_batch = tf.data.Dataset.from_tensor_slices(X_train).shuffle(5000).batch(BATCH_SIZE).take(1).get_single_element()

# Run the smoothed training step
d_loss_value, g_loss_value = train_step_smoothed(image_batch)

print(f"\nTraining Step Complete:")
print(f"D_Loss: {d_loss_value.numpy():.4f}")
print(f"G_Loss: {g_loss_value.numpy():.4f}")

# Example check
log_loss([d_loss_value.numpy()], [g_loss_value.numpy()])