# ML-Anomaly-Detection-in-Real-Time-Tracking-Data

## Overview

This project focuses on **anomaly detection in real-time tracking data** as part of the **Umbrella Project (UP-1)**. The goal is to establish a multi-layered, reliable anomaly detection framework using statistical methods and machine learning techniques. The project is divided into two main scripts:

1. **`UP1.py`**: Implements **statistical anomaly detection** using Z-scores and moving averages.
2. **`UP2.py`**: Implements **machine learning-based anomaly detection** using the Isolation Forest algorithm.

Both scripts include interactive dashboards for real-time visualization of anomalies.

---

## Files

### 1. `UP1.py`
- **Purpose**: Detects anomalies using statistical methods (Z-scores and moving averages).
- **Input**: `UP-1_Anomaly_Detection_SynData.csv` (synthetic tracking data).
- **Output**:
  - Console: Classification report and prediction accuracy.
  - Dashboard: Real-time visualization of anomalies.

### 2. `UP2.py`
- **Purpose**: Detects anomalies using the Isolation Forest algorithm.
- **Input**: `UP-1_Anomaly_Detection_SynData.csv` (synthetic tracking data).
- **Output**:
  - Console: Classification report and prediction accuracy.
  - Dashboard: Real-time visualization of anomalies with filtering options.

---

## How to Run

### Prerequisites
1. Install Python 3.x.
2. Install the required libraries:
   ```bash
   pip install pandas dash plotly scikit-learn scipy

### Run up1.py:
   ```
python up1.py
  ```
Access the dashboard at http://127.0.0.1:8050

### Run up2.py:
```
python up2.py
  ```
Access the dashboard at http://127.0.0.1:8050

## Results

### 1. `UP1.py`
#### Statistical Anomaly Detection:

- Uses Z-scores and moving averages to detect anomalies.

- Highlights anomalies in red (Z-score) and orange (moving average) on the dashboard.

- Provides a classification report and prediction accuracy in the console.

### 2. `UP2.py`
#### Machine Learning Anomaly Detection:

- Uses the Isolation Forest algorithm to detect anomalies.

- Allows filtering by location (latitude, longitude) and time (hour, day, month) on the dashboard.

- Provides a classification report and prediction accuracy in the console.
