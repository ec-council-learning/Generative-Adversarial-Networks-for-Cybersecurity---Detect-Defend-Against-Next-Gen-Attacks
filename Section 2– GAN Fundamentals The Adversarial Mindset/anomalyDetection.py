import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. Model Parameters and Rebuilding (Based on Video 1) ---
# ==============================================================================

LATENT_DIM = 32
INPUT_DIM = 784 # 28 * 28

# 1a. Data Preprocessing Utility
def preprocess_data(data):
    """Normalize and flatten the MNIST images."""
    data = data.astype('float32') / 255.0
    data = data.reshape((len(data), INPUT_DIM))
    return data

# 1b. Rebuild the Autoencoder Architecture
def build_autoencoder():
    """Defines the simple 784 -> 32 -> 784 Autoencoder model."""
    input_img = Input(shape=(INPUT_DIM,), name='Input_784_Pixels') 
    
    # Encoder
    encoded = Dense(128, activation='relu', name='Encoder_Hidden_1')(input_img)
    encoded = Dense(64, activation='relu', name='Encoder_Hidden_2')(encoded)
    latent_vector = Dense(LATENT_DIM, activation='relu', name='Latent_Vector_32_Features')(encoded) 

    # Decoder
    decoded = Dense(64, activation='relu', name='Decoder_Hidden_1')(latent_vector)
    decoded = Dense(128, activation='relu', name='Decoder_Hidden_2')(decoded)
    output_img = Dense(INPUT_DIM, activation='sigmoid', name='Output_Reconstruction_784_Pixels')(decoded) 
    
    autoencoder = Model(input_img, output_img, name='Full_Autoencoder_Model')
    return autoencoder

# ==============================================================================
# --- 2. Data Preparation for Anomaly Detection ---
# ==============================================================================

(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

# Define the "Normal" class (e.g., digit '1') and the "Anomaly" class (e.g., digit '5')
NORMAL_CLASS = 1
ANOMALY_CLASS = 5

# Create the training data (only the Normal Class)
x_train_normal = preprocess_data(x_train_full[y_train_full == NORMAL_CLASS])
# Create the test sets
x_test_normal = preprocess_data(x_test_full[y_test_full == NORMAL_CLASS])
x_test_anomaly = preprocess_data(x_test_full[y_test_full == ANOMALY_CLASS])

print(f"✅ Data Sets Prepared:")
print(f"   Training Data (Normal '{NORMAL_CLASS}'): {len(x_train_normal)} samples")
print(f"   Test Data (Anomaly '{ANOMALY_CLASS}'): {len(x_test_anomaly)} samples")

# ==============================================================================
# --- 3. Train the Autoencoder (Only on Normal Data) ---
# ==============================================================================

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

print("\n--- Training Autoencoder on NORMAL Data ('1's only) ---")
# Train only on the normal data, forcing the AE to only learn the pattern of '1'
autoencoder.fit(
    x_train_normal, x_train_normal,
    epochs=15, 
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_normal, x_test_normal),
    verbose=0 # Suppress training output for clean demonstration
)
print("✅ Training complete. Model is optimized for Normal ('1') patterns.")

# ==============================================================================
# --- 4. Anomaly Detection Logic and Threshold Setting ---
# ==============================================================================

# 4a. Predict and Calculate Reconstruction Error (MSE)
def calculate_errors(model, data):
    """Calculates the Reconstruction Error (MSE) for a given dataset."""
    reconstructions = model.predict(data, verbose=0)
    # The reconstruction error is the mean squared error across all 784 dimensions
    # for each sample.
    errors = np.mean(np.square(data - reconstructions), axis=1)
    return errors, reconstructions

# Calculate the error distribution on the NORMAL training data
normal_train_errors, _ = calculate_errors(autoencoder, x_train_normal)

# 4b. Set the Anomaly Threshold
# Use Mean + 3 Standard Deviations (a common starting point for k=3)
MEAN_ERROR = np.mean(normal_train_errors)
STD_DEV_ERROR = np.std(normal_train_errors)
K_SIGMA = 3.0
ANOMALY_THRESHOLD = MEAN_ERROR + (K_SIGMA * STD_DEV_ERROR)

print(f"\n--- Anomaly Detection Parameters ---")
print(f"   Mean Normal Error: {MEAN_ERROR:.6f}")
print(f"   Std Dev Normal Error: {STD_DEV_ERROR:.6f}")
print(f"   Anomaly Threshold (k={K_SIGMA}): {ANOMALY_THRESHOLD:.6f}")

# 4c. Test the System
test_normal_errors, test_normal_reconstructions = calculate_errors(autoencoder, x_test_normal)
test_anomaly_errors, test_anomaly_reconstructions = calculate_errors(autoencoder, x_test_anomaly)

# Count detections
normal_detections = np.sum(test_normal_errors > ANOMALY_THRESHOLD)
anomaly_detections = np.sum(test_anomaly_errors > ANOMALY_THRESHOLD)

print(f"\n--- Detection Results on Test Data ---")
print(f"   Normal Test Samples ('{NORMAL_CLASS}'): {len(x_test_normal)}")
print(f"   False Positives (Normal classified as Anomaly): {normal_detections} ({normal_detections/len(x_test_normal)*100:.2f}%)")
print(f"   Anomaly Test Samples ('{ANOMALY_CLASS}'): {len(x_test_anomaly)}")
print(f"   True Positives (Anomaly classified as Anomaly): {anomaly_detections} ({anomaly_detections/len(x_test_anomaly)*100:.2f}%)")


# ==============================================================================
# --- 5. Visualization: Proving the Principle ---
# ==============================================================================

def visualize_anomalies(anomaly_data, reconstructions, errors, threshold, n=5):
    """Plots original, reconstructed, and the error score for samples."""
    
    # Sort by error to pick the worst-reconstructed (most anomalous) samples
    sorted_indices = np.argsort(errors)[::-1]
    
    plt.figure(figsize=(20, 4))
    plt.suptitle(f"Anomaly Detection (Normal: '{NORMAL_CLASS}', Anomaly: '{ANOMALY_CLASS}'). Threshold: {threshold:.4f}", fontsize=16, y=1.05)
    
    for i in range(n):
        idx = sorted_indices[i]
        
        # --- 1. Display Original Anomaly Image ---
        ax1 = plt.subplot(2, n, i + 1)
        plt.imshow(anomaly_data[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Anomaly Input\nScore: {errors[idx]:.4f}")
        ax1.axis('off')

        # --- 2. Display Reconstructed (Failed) Image ---
        ax2 = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[idx].reshape(28, 28), cmap='gray')
        
        is_anomaly = errors[idx] > threshold
        color = 'red' if is_anomaly else 'green'
        
        plt.title(f"Reconstruction (Detected: {is_anomaly})", color=color)
        ax2.axis('off')

    plt.show()
    
# Visualize the 5 most clearly detected anomalies (digit '5')
print("\n--- Visualizing Detected Anomalies ---")
visualize_anomalies(x_test_anomaly, test_anomaly_reconstructions, test_anomaly_errors, ANOMALY_THRESHOLD, n=5)

# Visualize 5 normal samples for comparison (should have low error)
print("\n--- Visualizing Normal Samples (Control) ---")
# Pick 5 random normal test samples
np.random.seed(42)
random_indices = np.random.choice(len(x_test_normal), 5, replace=False)
x_normal_subset = x_test_normal[random_indices]
errors_subset, reconstructions_subset = calculate_errors(autoencoder, x_normal_subset)

visualize_anomalies(x_normal_subset, reconstructions_subset, errors_subset, ANOMALY_THRESHOLD, n=5)
# ==============================================================================
# --- 5. Summary Plotting Code (New Section) ---
# ==============================================================================

# Prepare data for the box plot
data_to_plot = [normal_train_errors, test_normal_errors, test_anomaly_errors]
labels = ['Normal Training', 'Normal Test', 'Anomaly Test']

plt.figure(figsize=(7, 6)) 
box_plot = plt.boxplot(data_to_plot, 
                       vert=True,
                       patch_artist=True,
                       labels=labels,
                       showfliers=True)

# Set colors for the boxes
colors = ['lightblue', 'lightgreen', 'salmon']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

# Plot the Anomaly Threshold line
plt.axhline(
    ANOMALY_THRESHOLD, 
    color='red', 
    linestyle='--', 
    linewidth=2, 
    label=f'Threshold ({ANOMALY_THRESHOLD:.4f})'
)

# Set labels and title
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Anomaly Detection: Error Distribution vs. Threshold')
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--')

# Adjust y-limit to focus on the separation
y_max = max(np.max(test_anomaly_errors), ANOMALY_THRESHOLD) * 1.1
plt.ylim(0, y_max)

# Save the plot
plt.savefig('anomaly_summary_boxplot.png')
print("\n✅ Summary plot 'anomaly_summary_boxplot.png' saved.")


# --- Original Visualization (Commented out for clean execution) ---
# The original visualization functions are kept, but the plt.show()
# calls are suppressed to prevent interactive window issues.

# def visualize_anomalies(...):
#     # ... (function body) ...
#     # plt.show() # Suppressed

# visualize_anomalies(...)
# visualize_anomalies(...)
