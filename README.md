Wind Turbine Gearbox Vibration Monitoring and Anomaly Detection Using Data Analytics

Project Description

This project is a Python-based Engineering Data Analytics System designed for monitoring wind turbine gearbox health using vibration analysis and machine learning techniques. The system processes gearbox telemetry data, performs statistical analytics, detects anomalies using machine learning, and generates visualization dashboards for predictive maintenance applications.

The project aims to improve gearbox reliability, reduce maintenance costs, and prevent unexpected failures through early fault detection and continuous condition monitoring.

The system analyzes multiple operational parameters including:

- Vibration X-axis
- Vibration Y-axis
- Vibration Z-axis
- Gearbox bearing temperature
- Oil pressure
- Particle count
- Timestamp data

An Isolation Forest machine learning algorithm is used to identify abnormal operating conditions associated with excessive vibration, imbalance, bearing wear, shaft misalignment, lubrication failure, and thermal irregularities.

---

Features

- Automated CSV telemetry data processing
- Data cleaning and preprocessing
- Statistical analysis and summary generation
- Correlation analysis
- Machine learning-based anomaly detection
- Static and interactive visualization dashboards
- Automated output file generation
- Predictive maintenance support

---
Technologies Used

Programming Language
- Python 3.14.2

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

---

System Workflow

1. Load gearbox telemetry dataset
2. Clean and preprocess data
3. Perform statistical analysis
4. Apply anomaly detection using Isolation Forest
5. Generate visualizations and dashboards
6. Export processed datasets and outputs

---

Project Structure

```bash
project/
│
├── data/
│   ├── gearbox_telemetry.csv
│
├── outputs/
│   ├── cleaned_dataset.csv
│   ├── statistical_summary.csv
│   ├── vibration_trends.png
│   ├── heatmap.png
│   ├── boxplot.png
│   ├── dashboard.html
│
├── src/
│   ├── preprocessing.py
│   ├── analytics.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   ├── main.py
│
├── requirements.txt
├── README.md
