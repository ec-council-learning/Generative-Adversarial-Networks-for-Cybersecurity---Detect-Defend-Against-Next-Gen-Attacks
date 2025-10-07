# ----------------------------------------------------------------------
# Video 2: Working with Data – Using NumPy and Pandas for Data Manipulation
# Focus: NumPy for fast array math (vectorization) and Pandas for data handling.
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd

# --- PART 1: NumPy (Numerical Foundation) ---

# 1. Creating an ndarray (N-dimensional Array)
# This is the fast, efficient data structure used by all AI/ML models.
# Example: A batch of 3 samples, each with 4 features (e.g., speed, size, protocol, port)
feature_matrix = np.array([
    [100, 5, 6, 443],
    [120, 8, 1, 80],
    [90, 4, 3, 22]
])
print("Original Feature Matrix:\n", feature_matrix)

# 2. Vectorization (Fast, Array-wide Math)
# Perform a calculation on every element simultaneously (e.g., scale all values by 0.5)
# This is much faster than using Python loops.
scaled_matrix = feature_matrix / 2.0
print("\nScaled Matrix (Vectorization):\n", scaled_matrix)

# 3. Shape and Dimensions
# The shape is critical for ensuring the data fits into your neural network architecture.
print(f"\nMatrix Shape (Rows, Cols): {scaled_matrix.shape}")
print("-" * 40)

# --- PART 2: Pandas (Data Handling) ---

# 4. Creating a DataFrame (Structured Data)
# A DataFrame is essential for handling real-world structured security data (e.g., CSV logs).
data = {
    'Timestamp': ['10:00', '10:01', '10:02', '10:03'],
    'Source_IP': ['192.168.1.1', '10.0.0.5', '192.168.1.1', '1.2.3.4'],
    'User_Agent': ['Chrome', 'Python/3.9', 'Chrome', 'Unknown'],
    'Status': ['200', '404', '200', '500'],
    'Bytes': [5000, 120, 5000, 8500],
}
log_df = pd.DataFrame(data)
print("Initial Data Frame:\n", log_df)

# 5. Data Filtering (Querying the Data)
# Filter the DataFrame to quickly query specific conditions (e.g., isolate successful connections).
success_logs = log_df[log_df['Status'] == '200']
print("\nFiltered Success Logs (Status 200):\n", success_logs)

# 6. Data Cleaning (Handling Missing/Bad Data)
# Example: Check the User_Agent column for any 'Unknown' entries—a common step in threat hunting.
unknown_user_agents = log_df[log_df['User_Agent'] == 'Unknown']
print(f"\nNumber of Unknown User Agents: {len(unknown_user_agents)}")