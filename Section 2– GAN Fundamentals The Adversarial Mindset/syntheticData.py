import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration Constants ---
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 200
NOISE_DIM = 100 # Defining the Noise Vector (z)
IMAGE_SHAPE = (28, 28, 1) # MNIST image size

# Set up output directory for generated images
if not os.path.exists('gan_generated_images'):
    os.makedirs('gan_generated_images')

# --- 1. Data Preparation for GANs (Scaling to [-1, 1]) ---
def preprocess_data():
    """Loads and normalizes MNIST data to the DCGAN's required [-1, 1] range."""
    # Load MNIST dataset
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshape and cast to float
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

    # Step 1: Normalize 0-255 -> 0.0-1.0
    train_images = train_images / 255.0

    # Step 2: Scale 0.0-1.0 -> -1.0-1.0 (via 2x - 1)
    train_images = (train_images * 2) - 1

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

# --- 2. The Generator Network ---
def build_generator():
    """
    Builds the DCGAN Generator model.
    Input: Random noise vector (z).
    Output: Synthetic image (scaled -1 to 1) using tanh activation.
    """
    # Use Sequential model
    model = Sequential(name='Generator')
    
    # Starts with a Dense layer to project the z vector into a small volume (7x7x256)
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Reshape to a 7x7x256 volume
    model.add(Reshape((7, 7, 256)))

    # Layer 1: Conv2DTranspose (7x7 -> 14x14)
    model.add(Conv2DTranspose(
        128, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        use_bias=False
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Layer 2: Conv2DTranspose (14x14 -> 28x28)
    # Output: Final Conv2D layer with tanh activation
    model.add(Conv2DTranspose(
        1, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        use_bias=False, 
        activation='tanh'
    ))

    return model

# --- 3. The Discriminator Network ---
def build_discriminator():
    """
    Builds the DCGAN Discriminator model.
    Input: Image (28x28x1).
    Output: Real/Fake probability score (sigmoid activation).
    """
    # Use Sequential model
    model = Sequential(name='Discriminator')

    # Layer 1: Conv2D with stride=2 for downsampling (28x28 -> 14x14)
    model.add(Conv2D(
        64, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        input_shape=IMAGE_SHAPE
    ))
    # Activation: All hidden layers use LeakyReLU(alpha=0.2)
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Layer 2: Conv2D (14x14 -> 7x7)
    model.add(Conv2D(
        128, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        use_bias=False
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Layer 3: Conv2D (7x7 -> 4x4)
    model.add(Conv2D(
        256, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        use_bias=False
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Flatten and output
    model.add(Flatten())
    # Output: Single neuron with sigmoid activation for Real/Fake probability
    model.add(Dense(1, activation='sigmoid'))

    return model

# --- 4. Define Loss Functions ---

# Use Binary Crossentropy Loss for both networks
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    # Loss for real images (should be 1s)
    # The smooth label 0.9 helps prevent D from becoming too confident too early
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output) 
    # Loss for fake images (should be 0s)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # Goal: G must get better at fooling D, so we label fake images as 1.0
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# --- 5. Adversarial Training Loop Steps ---

@tf.function
def train_discriminator(images, generator, discriminator, discriminator_optimizer):
    """Step 1: Training the Discriminator (D)"""
    
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    real_accuracy = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
    
    return disc_loss, real_accuracy, fake_accuracy


@tf.function
def train_generator(generator, discriminator, generator_optimizer):
    """Step 2: Training the Generator (G)"""
    
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        
        # Discriminator is NOT trained here
        fake_output = discriminator(generated_images, training=False) 
        
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return gen_loss

# --- 6. Visualization Utility ---
# Fixed noise vector for consistent visualization of progress
seed = tf.random.normal([16, NOISE_DIM]) 

def generate_and_save_images(model, epoch, test_input):
    """Visualizing Progress: Periodically save and plot the images generated by G."""
    # The training=False flag ensures layers like BatchNormalization run in inference mode
    predictions = model(test_input, training=False) 

    fig = plt.figure(figsize=(4, 4))
    
    # Rescale the images from [-1, 1] back to [0, 1] for plotting
    predictions = predictions * 0.5 + 0.5 

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'gan_generated_images/image_at_epoch_{epoch:04d}.png')
    plt.close()

# --- 7. Full Training Loop ---

def train_gan(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """The main training loop, running D and G training steps sequentially."""
    for epoch in range(epochs):
        disc_loss_list = []
        gen_loss_list = []
        real_acc_list = []
        fake_acc_list = []

        for image_batch in dataset:
            # Step 1: Train Discriminator (D)
            d_loss, r_acc, f_acc = train_discriminator(image_batch, generator, discriminator, discriminator_optimizer)
            disc_loss_list.append(d_loss.numpy())
            real_acc_list.append(r_acc.numpy())
            fake_acc_list.append(f_acc.numpy())
            
            # Step 2: Train Generator (G)
            g_loss = train_generator(generator, discriminator, generator_optimizer)
            gen_loss_list.append(g_loss.numpy())

        # Print progress and save images periodically
        avg_d_loss = np.mean(disc_loss_list)
        avg_g_loss = np.mean(gen_loss_list)
        # Average discriminator accuracy should be around 50-80% for healthy training
        avg_d_acc = np.mean([np.mean(real_acc_list), np.mean(fake_acc_list)])
        
        print(f"Epoch {epoch+1}/{epochs}: D_Loss={avg_d_loss:.4f}, G_Loss={avg_g_loss:.4f}, D_Acc={avg_d_acc:.4f}")
        
        # Save generated images for visualization
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

    # Generate final image after training
    # --- 8. Initialization and Execution ---

if __name__ == '__main__':
    # 1. Prepare Dataset
    train_dataset = preprocess_data()
    print("Dataset prepared and scaled to [-1, 1].")

    # 2. Build Models
    generator = build_generator()
    discriminator = build_discriminator()
    print("Generator and Discriminator models built.")

    # Optional: Print model summaries to see architecture
    # generator.summary()
    # discriminator.summary()

    # 3. Define Optimizers
    # DCGAN typically uses separate, balanced optimizers for each network.
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
    print("Optimizers (Adam with learning_rate=1e-4, beta_1=0.5) initialized.")

    # Generate initial image before training starts
    generate_and_save_images(generator, 0, seed)
    print("Saved initial random noise image (Epoch 0).")

    # 4. Start Training
    print(f"\nStarting DCGAN training for {EPOCHS} epochs...")
    train_gan(
        train_dataset, 
        EPOCHS, 
        generator, 
        discriminator, 
        generator_optimizer, 
        discriminator_optimizer
    )
    
    print("\nTraining complete! Generated images are saved in 'gan_generated_images' directory.")
    generate_and_save_images(generator, epochs, seed)

# --- Execute Script ---
