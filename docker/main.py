import os
import json
import subprocess
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import uvicorn
import tensorflow
import keras

app = FastAPI()

# Set dynamic file paths (use container paths)
LOGS_DIR = os.getenv("LOGS_DIR", "/app/logs")
# METRICS_FILE = os.path.join(LOGS_DIR, "model_metrics.json")
METRICS_FILE_PATH = "/app/logs/model_metrics.json"
LOG_FILE = os.path.join(LOGS_DIR, "ml_model.log")

@app.get("/")
async def root():
    return {"message": "ML Monitoring API is running!"}

# Prometheus metrics setup
registry = CollectorRegistry()
rmse_gauge = Gauge('n_rmse', 'Root Mean Squared Error', registry=registry)
accuracy_gauge = Gauge('accuracy', 'Model Accuracy', registry=registry)

@app.get("/metrics")
def get_metrics():
    # Debugging: Print where FastAPI is looking for the file
    print(f"🔍 Checking metrics file at: {METRICS_FILE_PATH}")

    # Verify if file exists
    if not os.path.exists(METRICS_FILE_PATH):
        return {"error": f"Metrics file not found at {METRICS_FILE_PATH}!"}
    
    # Read the file
    with open(METRICS_FILE_PATH, "r") as f:
        data = f.read()
    
    return {"metrics": data}

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"error": "Log file not found!"}
    with open(LOG_FILE, "r") as f:
        logs = f.readlines()[-10:]
    return {"logs": logs}

def log_streamer():
    if not os.path.exists(LOG_FILE):
        yield "Log file not found!"
        return
    with open(LOG_FILE, "r") as f:
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(1)

@app.get("/stream_logs")
def stream_logs():
    return StreamingResponse(log_streamer(), media_type="text/plain")

# Analyze performance using Llama 3.2 by running the analysis script
@app.get("/analyze")
async def analyze():
    try:
        result = subprocess.run(
            ["python3", "/app/analyze_performance_automate.py"],
            capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Analysis script timed out.")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Error: {result.stderr}")
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode
    }

# Trigger retraining by running the retrain script
def retrain_model():
    subprocess.run(["python3", "/app/retrain_model.py"])

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"message": "Retraining started in the background!"}

# Lifespan event to load the model at startup (if available)
async def lifespan(app: FastAPI):
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/retrained_model.h5")
    if not os.path.exists(model_path):
        print("⚠️ Model file not found! Skipping model load.")
    else:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    yield
    print("🔄 Shutting down server...")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
