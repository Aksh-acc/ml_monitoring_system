import os
import json
import math
import time
import logging
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ✅ Define Directories
LOGS_DIR = os.getenv("LOGS_DIR", "/app/logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
CSV_PATH = os.getenv("CSV_PATH", "/app/data/Iris.csv")

# ✅ Log File Setup
LOG_FILE = os.path.join(LOGS_DIR, "ml_model.log")

# ✅ Ensure Directories Exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Configure Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Function to log real events
def log_event(level, message):
    """Logs an event in JSON format."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if level.lower() == "info":
        logging.info(message)
    elif level.lower() == "error":
        logging.error(message)

# ✅ Load Dataset Safely
if not os.path.exists(CSV_PATH):
    log_event("error", f"Dataset not found: {CSV_PATH}")
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# ✅ Assume the last column is the label (species), rest are features
features = df.iloc[:, :-1].values  # All columns except the last
labels = df.iloc[:, -1].values  # Last column (species names)

# ✅ Encode categorical labels as numbers
encoder = OneHotEncoder(sparse_output=False)
labels_encoded = encoder.fit_transform(labels.reshape(-1, 1))

# ✅ Standardize feature values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# ✅ Define Neural Network Model
def create_model():
    log_event("info", "Creating new ML model...")
    model = keras.Sequential([
        Dense(100, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(50, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")  # Output layer should match class count
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["mae", "mse"]
    )
    return model

# ✅ Retrain the model
def retrain():
    log_event("info", "Starting ML model retraining...")

    model = create_model()
    log_event("info", "Training started...")

    history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

    # ✅ Save Model with Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"retrained_model_{timestamp}.h5")
    model.save(model_path)
    log_event("info", f"✅ Model retrained and saved: {model_path}")

    # ✅ Evaluate & Log Results
    loss, mae, mse = model.evaluate(X_train, y_train, verbose=0)
    rmse = math.sqrt(mse)
    log_event("info", f"Model evaluated. RMSE: {rmse:.4f}, MAE: {mae:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    retrain()
