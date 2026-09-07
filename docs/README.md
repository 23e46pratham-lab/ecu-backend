# ECU Guardian Documentation Hub

Welcome to the **ECU Guardian** technical documentation library. This directory contains detailed guides, model architecture documents, and end-to-end system explanations for developers, data scientists, and evaluators.

---

## Documentation Index

### 1. [End-to-End System Architecture & Workflow Guide](file:///e:/MP-ECU/ecu-backend/docs/end_to_end_system_architecture.md)
A comprehensive explanation of how the whole platform works, including:
- Dual-subsystem integration (OBD-II simulation + ML inference).
- Complete architectural decomposition and state diagrams.
- Data ingestion, cleaning, normalization, and WebSocket streaming.
- Complete API route map and data flow sequence diagrams.
- Hardware transition roadmap (ELM327 / CAN-bus).

### 2. [K-Means Driver Behavior Profiling Model](file:///e:/MP-ECU/ecu-backend/docs/kmeans_driver_behavior.md)
Detailed specification for `models/kmeans_driver_behavior_model.joblib` and `models/standard_scaler_driver_behavior.joblib`:
- Feature engineering over rolling time-series windows (std, MAD, acceleration range).
- Cluster interpretation (`Economical`, `Moderate`, `Harsh`).
- Mathematical formulations and data preprocessing.
- API payload schema and error handling.

### 3. [Random Forest Vehicle Health Classification Model](file:///e:/MP-ECU/ecu-backend/docs/random_forest_vehicle_health.md)
Detailed specification for `models/random_forest_health.pkl` and `models/scaler_health.pkl`:
- 8-sensor powertrain telemetry feature space.
- Random forest ensemble decision logic, bagging, and probability estimation.
- Health status categorization (`Normal`, `Warning`, `Critical`) and confidence metrics.
- Sub-millisecond serving via `POST /api/health/predict`.

### 4. [Hybrid LSTM + Random Forest Driver Behavior Pipeline](file:///e:/MP-ECU/ecu-backend/docs/driver_behaviour_lstm_rf.md)
Next-generation architectural roadmap:
- Two-stage hybrid pipeline combining deep learning sequence modelling (LSTM) with interpretable ensemble classification (Random Forest).
- Comparison against the baseline K-Means approach.
- Planned artifact dependencies and integration steps.

---

## Directory Overview

```
docs/
├── README.md                           # This index file
├── end_to_end_system_architecture.md   # Complete system workflow and architecture
├── kmeans_driver_behavior.md           # K-Means driver profiling model doc
├── random_forest_vehicle_health.md     # Random Forest health classifier doc
└── driver_behaviour_lstm_rf.md         # Next-gen LSTM + RF hybrid pipeline doc
```
