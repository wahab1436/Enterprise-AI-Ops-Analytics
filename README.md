# Enterprise AIOps Dashboard

A production-ready AIOps dashboard for analyzing incidents, forecasting system load, and detecting anomalies using machine learning and operational telemetry data.

## Features
- Upload and validate operational datasets  
- Automated data cleaning and preprocessing  
- Feature engineering for time, system, and workload metrics  
- Incident classification with XGBoost  
- CPU load forecasting using ML models  
- Anomaly detection with statistical thresholds  
- KPI panels, trends, heatmaps, and system-level insights  
- Exportable reports and processed datasets  

## Data Requirements

The dataset should include the following columns:

| Column | Description |
|--------|-------------|
| timestamp | Datetime of record |
| system_id | Unique system identifier |
| incident_type | Type/category of incident |
| priority | Severity level |
| cpu_usage | CPU load percentage |
| memory_usage | Memory consumption |
| disk_io | Disk throughput |
| network_traffic | Network usage |
| change_count | Configuration changes count |
| tickets_opened | Number of opened tickets |
| resolution_time | Time taken to resolve the incident |

Recommended minimum size: **10,000+ rows**.

## Architecture
Upload → Clean → Feature Engineering → ML Models → Visualizations → Export

perl
Copy code

## Installation
```bash
pip install -r requirements.txt
Run
bash
Copy code
streamlit run aiops_dashboard.py
Tech Stack
Python

Streamlit

Pandas, NumPy

Scikit-learn

XGBoost

SHAP

Plotly

Outputs
Cleaned dataset

Feature-engineered dataset

Incident classification predictions

CPU forecasts and anomaly flags

Evaluation metrics and reports
