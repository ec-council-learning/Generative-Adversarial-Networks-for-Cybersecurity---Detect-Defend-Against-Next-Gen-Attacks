# --- 3. DCGAN Training Implementation (Continuing from Video 2) ---

# Define Optimizers and Losses
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Compile Discriminator (standalone)
discriminator.compile(optimizer=d_optimizer, loss=loss_fn, metrics=['accuracy'])
discriminator.trainable = False # Important: Freeze D's weights when training G

# Build the Adversarial Model (Combined G and D)
# The input is the latent noise, and the output is D's classification of G's image
z_input = Input(shape=(LATENT_DIM,))
g_output = generator(z_input)
d_output = discriminator(g_output)
gan = Model(z_input, d_output, name='gan')
gan.compile(optimizer=g_optimizer, loss=loss_fn) 


# --- Placeholder for Data Loading (Assume X_train is pre-processed to [-1, 1])
# X_train = load_and_preprocess_image_data_in_range_minus_1_to_1(...) 
# For demonstration, use a dummy array
X_train = np.random.uniform(-1, 1, size=(5000, 64, 64, 3)) 
BATCH_SIZE = 128
EPOCHS = 100 

# --- The Training Loop (Simplified for clarity) ---
@tf.function
def train_step(images):
    # 1. Train Discriminator (Real samples)
    real_labels = tf.ones((BATCH_SIZE, 1))
    with tf.GradientTape() as tape:
        d_loss_real = loss_fn(real_labels, discriminator(images, training=True))
    d_gradients_real = tape.gradient(d_loss_real, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients_real, discriminator.trainable_variables))

    # 2. Train Discriminator (Fake samples)
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    fake_images = generator(noise, training=False)
    fake_labels = tf.zeros((BATCH_SIZE, 1))
    with tf.GradientTape() as tape:
        d_loss_fake = loss_fn(fake_labels, discriminator(fake_images, training=True))
    d_gradients_fake = tape.gradient(d_loss_fake, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients_fake, discriminator.trainable_variables))

    d_loss = d_loss_real + d_loss_fake

    # 3. Train Generator (The GAN Model)
    misleading_labels = tf.ones((BATCH_SIZE, 1)) # G wants D to output 'Real' (1)
    with tf.GradientTape() as tape:
        g_loss = gan(noise, training=True)
    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, g_loss

# Example Training Loop Call (This would be run for EPOCHS)
# for epoch in range(EPOCHS):
#     for image_batch in tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000).batch(BATCH_SIZE):
#         d_loss_value, g_loss_value = train_step(image_batch)
#     print(f"Epoch {epoch+1}, D_Loss: {d_loss_value.numpy():.4f}, G_Loss: {g_loss_value.numpy():.4f}")


# --- Anomaly Detection Setup (Video 4 Lab) ---
# After training the GAN for many epochs...

def calculate_anomaly_score(real_sample, generator, discriminator):
    """
    Anomaly Score is the sum of:
    1. Reconstruction Error (L2 distance in data space) - requires a different model like an AE, 
       or optimizing Z for the closest match.
    2. Discriminator Confidence (D(x)) - How "real" the Discriminator thinks the sample is.
    
    For a simpler, initial lab using ONLY the trained D: 
    A low D(x) for a real sample implies it's *not* like the training data.
    """
    # Discriminator Confidence Score
    # Sample must be pre-processed the same way as training data (e.g., scaled to [-1, 1])
    d_output = discriminator(tf.expand_dims(real_sample, axis=0))
    # Anomaly: A low score (close to 0) means D thinks it's FAKE, i.e., an anomaly from P_data.
    return 1.0 - d_output.numpy()[0][0] # Score: 1-D(x) so higher score = higher anomaly

# Example Usage:
# benign_sample = X_train[0] # D should think this is real (score close to 0)
# anomaly_sample = np.random.uniform(-5, 5, size=(64, 64, 3)) # D should think this is fake (score close to 1)

# print(f"Benign Anomaly Score: {calculate_anomaly_score(benign_sample, generator, discriminator):.4f}")
# print(f"Anomaly Anomaly Score: {calculate_anomaly_score(anomaly_sample, generator, discriminator):.4f}")