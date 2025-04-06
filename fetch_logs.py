import requests
import json

LOKI_URL = "http://localhost:3100"
LOKI_QUERY = '{job="ml_model"}'
LOG_FILE = "C:/Users/91966/ml_monitor/logs/ml_model.log"

def fetch_logs():
    response = requests.get(
        f"{LOKI_URL}/loki/api/v1/query_range",
        params={"query": LOKI_QUERY, "limit": 10}
    )

    logs = response.json()
    log_entries = []

    for stream in logs.get("data", {}).get("result", []):
        for entry in stream["values"]:
            timestamp, log_message = entry
            log_entries.append({"timestamp": timestamp, "message": log_message})

    with open(LOG_FILE, "w") as f:
        json.dump(log_entries, f, indent=4)

    return log_entries

if __name__ == "__main__":
    print(fetch_logs())
