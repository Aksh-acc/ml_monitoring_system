# ML Monitoring System with Real-Time Retraining & Observability

Welcome to the **ML Monitoring System** – an intelligent pipeline for monitoring, analyzing, and retraining ML models in real-time using Prometheus, Grafana, Loki, and LLaMA 3.2 with FastAPI integration.

> **Built by:** Aksh - [GitHub Portfolio](https://github.com/Aksh-acc) 🔗  
> **Status:** Actively maintained | Open to contributions 🤝

---

## 🚀 Features

- 📊 **Metrics Monitoring** using **Prometheus**
- 📁 **Log Collection** via **Promtail + Loki**
- 📈 **Visualization Dashboards** with **Grafana**
- 🤖 **AI-Based Analysis & Auto-Retraining** via **LLaMA 3.2**
- 🌐 **FastAPI** with endpoints to access logs, metrics, analysis, and retraining
- 🔁 Supports custom ML models for monitoring & feedback loops

---

## 🐳 Setup & Run

### 1. Clone the Repository
```bash
git clone https://github.com/Aksh-acc/ml_monitoring_system.git
cd ml_monitoring_system
```

### 2. Start All Services Using Docker
```bash
docker-compose up --build
```
Ensure all containers (Prometheus, Grafana, Loki) are running.

### 3. Run the Default ML Model
```bash
python ML_iris_model.py
```
This pushes metrics (accuracy, RMSE) and logs into the monitoring system.

### 4. Want to Use Your Own Model?
- Modify your custom model to push logs and metrics.
- Logs should be saved in `./logs`
- Metrics should include `accuracy`, `rmse` or custom ones.

Use `fetch_logs.py` to extract logs from storage.

### 5. Let LLaMA 3.2 Analyze and Improve
- LLaMA is integrated via `ollama` and monitors model health.
- If accuracy drops or over/underfitting is detected, it:
  - Adjusts hyperparameters (like epochs, learning rate)
  - Triggers automatic retraining
  - Saves new model in `models/` as `.h5`

### 6. FastAPI Endpoints

| Endpoint       | Functionality                            |
|----------------|------------------------------------------|
| `/metrics`     | Returns current model metrics            |
| `/logs`        | Fetches stored logs                     |
| `/analyze`     | Uses LLaMA to analyze model performance |
| `/retrain`     | Triggers model retraining               |

---

## Preview 

| Metrics Visualization (RMSE)           | Metrics Visualization (Accuracy)            |
|----------------------------------------|---------------------------------------------|
| ![Docker](assets/Screenshot%20(251).png) | ![Promtail Logs](assets/Screenshot%20(252).png) |

| Loki Dashboard                         | Prometheus (Targets)                        |
|----------------------------------------|---------------------------------------------|
| ![Model Stats](assets/Screenshot%20(253).png) | ![Grafana](assets/Screenshot%20(254).png) |

| Llama analysis result                  | PushGateway (Metrics)                       |
|----------------------------------------|---------------------------------------------|
| ![API Logs](assets/Screenshot%20(257).png) | ![FastAPI](assets/Screenshot%20(347).png) |

---

## 🎥 Video Tutorials

- [📈 Monitoring, Retraining & Automation](https://www.youtube.com/watch?v=hdWUDdzNtTA)
- [🚀 FastAPI Walkthrough](https://www.youtube.com/watch?v=RP2fcOHty1g)

---

## 💬 Description

I am a second-year engineering student pursuing **B.Tech in CSE (Data Science)** at **Manipal University Jaipur**. Passionate about AI, ML, and automation — I develop intelligent systems that improve with time.

> This system is a blend of AI observability, real-time feedback loops, and self-healing model automation.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📬 Contact

Let’s collaborate or geek out over ML! Reach me via:
- 🔗 GitHub: [Aksh-acc](https://github.com/Aksh-acc)
- 📧 LinkedIn: [Aksh Modi](https://www.linkedin.com/in/aksh-modi-9286a8289/)

---

## ⭐ Show Some Love
If this project helped you or inspired something awesome, please ⭐ star the repo and spread the word!

---

Built with 💙 using Python, Docker, and a passion for ML.

