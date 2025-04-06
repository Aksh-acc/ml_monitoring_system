import json
import re
import subprocess
import time
import ollama
import requests
import numpy as np

LOG_FILE = "/app/logs/ml_model.log"
METRICS_FILE = "/app/logs/model_metrics.json"
FASTAPI_RETRAIN_URL = "http://127.0.0.1:8080/retrain"

# ✅ Function to parse logs correctly
def parse_log_line(line):
    """Parses structured & unstructured log entries."""
    try:
        return json.loads(line.strip())  # If it's valid JSON, return it
    except json.JSONDecodeError:
        pass  # If not JSON, continue processing

    # ✅ Handle unstructured log format
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+) - (.+)"
    match = re.match(pattern, line.strip())

    if match:
        timestamp, level, message = match.groups()

        # ✅ Extract RMSE & MAE if they exist
        rmse_match = re.search(r"RMSE=([\d.]+)", message)
        mae_match = re.search(r"MAE=([\d.]+)", message)

        structured_log = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "RMSE": float(rmse_match.group(1)) if rmse_match else None,
            "MAE": float(mae_match.group(1)) if mae_match else None,
        }
        return structured_log  
    return None  # Return None if no match

# ✅ Load logs & metrics
def load_data():
    logs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parsed_log = parse_log_line(line)
            if parsed_log:
                logs.append(parsed_log)

    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError:
        print("Warning: Metrics file is corrupted or empty. Using default values.")
        metrics = {"accuracy": 0, "loss": float("inf")}

    return logs, metrics

# ✅ Analyze performance using Llama 3.2
def analyze_performance():
    logs, metrics = load_data()

    # Extract recent RMSE & MAE values
    rmse_values = [log.get("RMSE") for log in logs if log.get("RMSE") is not None]
    mae_values = [log.get("MAE") for log in logs if log.get("MAE") is not None]

    if not rmse_values:
        print(" No RMSE values found in logs!")
        return {"error": "No RMSE values found in logs."}

    # Compute averages safely
    avg_rmse = np.mean(rmse_values[-10:]) if len(rmse_values) >= 10 else np.mean(rmse_values)
    avg_mae = np.mean(mae_values[-10:]) if len(mae_values) >= 10 else np.mean(mae_values)

    # ✅ Ask Llama 3.2 for decision-making
    prompt = f"""
    The ML model's recent performance:
    - Average RMSE: {avg_rmse:.5f}
    - Average MAE: {avg_mae:.5f}

    Based on these values:
    1. If RMSE > 0.1 or MAE > 0.02, suggest retraining steps.
    2. Otherwise, confirm that the model is performing well.

    Give a clear decision: "YES" for retrain, "NO" for no retraining, and reasons.
    """

    response = ollama.chat("llama3.2", messages=[{"role": "user", "content": prompt}])
    analysis_text = response["message"]["content"].strip()

    return analysis_text

# ✅ Trigger retraining if needed
def check_and_retrain():
    analysis = analyze_performance()
    print(" Llama Analysis:", analysis)

    if "YES" in analysis:
        print("Triggering retraining via FastAPI!")
        response = requests.post(FASTAPI_RETRAIN_URL)
        print("Retrain Response:", response.text)
    else:
        print("Model is performing well, no retraining needed.")

# ✅ Run every hour
if __name__ == "__main__":
    check_and_retrain()
