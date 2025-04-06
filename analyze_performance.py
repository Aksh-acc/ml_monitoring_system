import json
import ollama
from fetch_logs import fetch_logs
from fetch_metrics import get_model_metrics

def analyze_performance():
    logs = fetch_logs()
    metrics = get_model_metrics()

    log_texts = "\n".join([f"{log['timestamp']} - {log['message']}" for log in logs])
    metric_texts = "\n".join([f"{key}: {value}" for key, value in metrics.items()])

    prompt = f"""
    The following logs and metrics are collected from an ML model:

    Logs:
    {log_texts}

    Metrics:
    {metric_texts}

    Analyze the ML model performance. Is it degrading? If yes, suggest improvements.
    """

    response = ollama.chat("llama3.2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

if __name__ == "__main__":
    print(analyze_performance())
