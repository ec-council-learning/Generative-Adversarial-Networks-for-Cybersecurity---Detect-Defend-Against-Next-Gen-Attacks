# ----------------------------------------------------------------------
# Video 3: Visualizing Data – Introduction to Matplotlib and Seaborn
# Focus: Creating basic plots and advanced plots (heatmaps/KDE) to spot anomalies.
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for better charts
sns.set_style("whitegrid") 

# --- Data Preparation ---
# Create synthetic data representing 'session duration' for two user groups
np.random.seed(42)
normal_sessions = np.random.normal(loc=60, scale=15, size=1000) # Mean 60 minutes
anomalous_sessions = np.random.normal(loc=180, scale=10, size=50) # Anomalies around 180 mins
session_data = np.concatenate([normal_sessions, anomalous_sessions])
session_data = session_data[session_data > 0] # Remove negative values

# --- PART 1: Matplotlib (Basic Plotting) ---

# 1. Basic Histogram to view distribution
# This shows raw event frequency—a fast way to check for basic outliers.
plt.figure(figsize=(8, 4))
plt.hist(session_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Session Duration (Raw Counts)')
plt.xlabel('Duration (Minutes)')
plt.ylabel('Number of Sessions')
plt.show()

# --- PART 2: Seaborn (Advanced Statistical Plots) ---

# 2. Kernel Density Estimate (KDE) Plot
# Smooths the distribution to better identify underlying clusters/anomalies (the "normal" baseline).
plt.figure(figsize=(8, 4))
sns.kdeplot(session_data, fill=True, color='red', alpha=0.5)
plt.title('Session Duration Density (Clearly Spotting Anomalies)')
plt.xlabel('Duration (Minutes)')
plt.ylabel('Density')
plt.show()

# 3. Scatter Plot for Outliers
# Combine session data with a simple 'score' to highlight outliers.
# This mimics analyzing two features simultaneously (e.g., session length vs. data uploaded)
upload_score = np.random.rand(len(session_data)) * 100
upload_score[normal_sessions.size:] += 200  # Give anomalies a higher score for visual separation

plt.figure(figsize=(8, 4))
# Use color mapping to instantly highlight the abnormal cluster (duration > 150 min)
plt.scatter(session_data, upload_score, alpha=0.6, 
            c=(session_data > 150), cmap='coolwarm') 
plt.title('Session Duration vs. Upload Score (Identifying Outliers)')
plt.xlabel('Session Duration (Minutes)')
plt.ylabel('Upload Score')
plt.show()
