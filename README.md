# 🚀 Enterprise AIOps Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 💡 Overview
The **Enterprise AIOps Analytics Platform** is an AI-powered solution designed to provide **intelligent incident prediction, anomaly detection, and operations insights** for IT systems. It empowers enterprises to optimize IT operations, proactively manage incidents, and make data-driven decisions in real-time.

---

## 🛠 Features
- **Data Upload & Processing**: Upload CSV telemetry data and perform automated cleaning, feature engineering, and aggregation.
- **Analytics Dashboard**: Visualize key system metrics, incident distributions, and temporal patterns.
- **Machine Learning Predictions**:
  - Incident classification
  - Workload regression
  - Real-time anomaly detection
- **Live Monitoring**: Generate live predictions for high-risk incidents and anomalies.
- **Export Center**: Download cleaned datasets, feature sets, predictions, and model performance reports.

---

##Data Requirements

Format: CSV

Columns: timestamp, system_id, metrics, incident_type, priority

Encoding: UTF-8

Size: ≤ 200 MB

##Machine Learning

Classification Model: Predict system incidents (Incident / No Incident)

Regression Model: Predict workload metrics (CPU, Memory, Disk I/O, Network Traffic)

Anomaly Detection: Identify outliers in real-time telemetry
