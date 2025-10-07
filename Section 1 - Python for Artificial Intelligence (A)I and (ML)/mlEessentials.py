# ------------------------------------------------------------
# Video 4: ML Essentials – Using Scikit-learn for Classification
# Focus: Data splitting, model training, evaluation, and prediction.
# ------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# --- 1. Load Dataset ---
# Iris dataset is a classic for classification tasks.
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Optional: Convert to DataFrame for inspection
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print("Sample of the dataset:\n", df.head())

# --- 2. Preprocess Features ---
# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Split Data ---
# Train-test split to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --- 4. Train Model ---
# Random Forest is robust and easy to interpret
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluate Model ---
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# --- 6. Make Predictions ---
# Predict on a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nPredicted class for sample:", target_names[prediction[0]])