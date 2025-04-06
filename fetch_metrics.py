from prometheus_api_client import PrometheusConnect
import json

PROMETHEUS_URL = "http://localhost:9090"  # Adjust if needed

def get_model_metrics():
    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

    # Define Prometheus queries for metrics
    queries = {
        "accuracy": 'accuracy',  
        "loss": 'n_rmse',  
    }

    results = {}
    for metric, query in queries.items():
        try:
            response = prom.custom_query(query=query)
            if response:
                results[metric] = float(response[0]['value'][1])  # Extract latest value
        except Exception as e:
            results[metric] = None
            print(f"Error fetching {metric}: {e}")

    # Save to JSON for LlamaIndex
    with open("C:/Users/91966/ml_monitor/logs/model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

# Test fetching metrics
if __name__ == "__main__":
    print(get_model_metrics())
