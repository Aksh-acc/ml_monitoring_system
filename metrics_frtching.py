import requests

PROMETHEUS_URL = "http://localhost:9090"  # Change based on your setup
QUERIES = {
    "rmse": 'n_rmse',  # Adjust metric name based on your setup
    "accuracy": 'accuracy'
}

def fetch_metrics():
    metrics_data = {}
    for key, query in QUERIES.items():
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        data = response.json()
        
        if data["status"] == "success":
            metrics_data[key] = float(data["data"]["result"][0]["value"][1])
        else:
            print(f"Failed to fetch {key}")
    
    return metrics_data

metrics = fetch_metrics()
print(metrics)  # Example Output: {'rmse': 12.5, 'accuracy': 0.72}
