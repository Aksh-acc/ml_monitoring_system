import numpy as np
from tensorflow import keras
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
LOG_FILE = "C:/Users/91966/ml_monitor/logs/ml_model.log"

# Function to log predictions
def log_event(level, message):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ✅ Define custom objects for loading
custom_objects = {"MeanSquaredError": keras.losses.MeanSquaredError()}

# ✅ Load the model correctly
try:
    model = keras.models.load_model(
        "C:/Users/91966/ml_monitor/models/retrained_model.h5",
        custom_objects=custom_objects
    )
    print("✅ Model loaded successfully!")
    log_event("info", "Model loaded successfully.")
except Exception as e:
    log_event("error", f"Failed to load model: {str(e)}")


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("✅ Model compiled successfully!")
csv_path = r"C:\Users\91966\ml_monitor\data\Iris.csv"
df = pd.read_csv(csv_path)

# ✅ Extract feature columns (first 4 columns)
features = df.iloc[:, :-1].values  

# ✅ Standardize using the training scaler
scaler = StandardScaler()
scaler.fit(features)
# Sample input for testing
raw_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  

# ✅ Standardize using the same scaler
features_scaled = scaler.fit_transform(raw_sample)

try:
    prediction = model.predict(features_scaled)
    species = ["setosa", "versicolor", "virginica"]
    predicted_label = species[np.argmax(prediction)]
    print("Predicted Class:", predicted_label)   
except Exception as e:
    log_event("error", f"Prediction failed: {str(e)}")

