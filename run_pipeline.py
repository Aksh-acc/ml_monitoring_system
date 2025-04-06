import os
from analyze_performance import analyze_performance

# Run the pipeline
os.system("python fetch_metrics.py")
analysis = analyze_performance()

if "performance degrading" in analysis.lower():
    print("⚠️ Model needs retraining! Starting process...")
    os.system("python retrain_model.py")
else:
    print("✅ Model is performing well. No retraining needed.")
