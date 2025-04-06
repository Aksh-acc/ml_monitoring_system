import fastapi
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import subprocess
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import ollama  # Llama 3.2 API

app = FastAPI()

# ✅ Prometheus metrics setup
registry = CollectorRegistry()
rmse_gauge = Gauge('n_rmse', 'Root Mean Squared Error', registry=registry)
accuracy_gauge = Gauge('accuracy', 'Model Accuracy', registry=registry)

# ✅ Fetch latest model metrics
@app.get("/metrics")
def get_metrics():
    with open("C:/Users/91966/ml_monitor/logs/model_metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics

# ✅ Fetch latest logs
@app.get("/logs")
def get_logs():
    with open("C:/Users/91966/ml_monitor/logs/ml_model.log", "r") as f:
        logs = f.readlines()[-10:]  # Get last 10 logs
    return {"logs": logs}

# ✅ Stream logs in real-time
def log_streamer():
    with open("C:/Users/91966/ml_monitor/logs/ml_model.log", "r") as f:
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(1)

@app.get("/stream_logs")
def stream_logs():
    return StreamingResponse(log_streamer(), media_type="text/plain")

# ✅ Analyze performance using Llama 3.2
@app.get("/analyze")
def analyze_performance():
    logs, metrics = get_logs(), get_metrics()
    prompt = f"""
    Given the following logs: {logs}
    and model metrics: {metrics},
    determine if the ML model performance is degrading. Suggest improvements.
    """
    response = ollama.chat("llama3", messages=[{"role": "user", "content": prompt}])
    return {"analysis": response["message"]["content"]}

# ✅ Automatically retrain model if performance drops
def retrain_model():
    subprocess.run(["python", "C:/Users/91966/ml_monitor/retrain_model.py"])

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"message": "Retraining started in the background!"}

# ✅ Auto-check performance and retrain if needed
@app.get("/auto_retrain")
def check_and_retrain():
    analysis = analyze_performance()
    
    if "degrading" in analysis["analysis"]:
        retrain_model()
        return {"message": "Model was degrading, retraining triggered!"}
    
    return {"message": "Model performance is fine. No retraining needed."}
