# Autovue: End-to-End System Architecture & Workflow Guide

This document provides an exhaustive, end-to-end technical explanation of how the **ECU Guardian Backend** operates. It covers system motivation, architecture, data flow, telemetry simulation, machine learning inference engines, real-time streaming, and API protocols.

---

## 1. System Mission & Core Philosophy

**AutoVue** is an automotive telemetry and predictive intelligence platform developed under the *Smart Vehicle ECU Monitoring and Predictive Maintenance* project (VTU, Dept. of ICBS, St Joseph Engineering College).

### The Problem
Traditional automotive telemetry development faces two major hurdles:
1. **Hardware Dependence**: Testing machine learning models or frontend dashboards requires physical access to a vehicle and an OBD-II dongle (ELM327) or expensive hardware-in-the-loop (HIL) test benches.
2. **Disconnected Diagnostics**: Existing commercial OBD-II scanners only report Diagnostic Trouble Codes (DTCs) after a catastrophic threshold breach has already occurred, offering little real-time assessment of driver stress or early mechanical degradation.

### The Unified Solution
AutoVue combines two core subsystems into a single unified, deployable FastAPI application:
1. **Real-Time OBD-II Simulator**: Replays high-resolution, multi-sensor driving datasets (CSV/XLSX) row-by-row with controllable playback speed ($0.5\times$ to $10\times$), looping, pausing, and live WebSocket broadcasting.
2. **Machine Learning Intelligence Hub**: Stateless, low-latency endpoints that ingest sensor telemetry to classify:
   - **Driver Behavior**: Profiling aggression levels (`Economical`, `Moderate`, `Harsh`) over rolling windows.
   - **Powertrain Health**: Multi-sensor anomaly detection (`Normal`, `Warning`, `Critical`) with probability confidence scores.

By unifying both components into a single process, the application avoids multi-container cold starts on hosting providers (like Render or Railway) and presents a single, cohesive API surface to client dashboards.

---

## 2. Global Architecture Diagram

The high-level system decomposition separates concerns between network transport, simulation engines, and machine learning inference while maintaining clean modularity.

```mermaid
flowchart TD
    subgraph ClientLayer["Clients & Frontend Presentation"]
        DASH["Built-in Simulator Dashboard (dashboard.html)"]
        WEBAPP["External Client / Web / Mobile App"]
        SWAGGER["FastAPI Interactive Docs (/docs, /redoc)"]
    end

    subgraph FastAPIServer["Unified FastAPI Application (app/main.py)"]
        ROUTER_SYS["System Routes: /health, /"]
        ROUTER_ML["ML Inference Router: /api/driver/*, /api/health/*"]
        ROUTER_SIM["Simulator REST Router: /api/start, /api/status, /api/datasets..."]
        WS_LIVE["WebSocket Channel: /api/ws/live (Telemetry Stream)"]
        WS_LOGS["WebSocket Channel: /api/ws/logs (Simulator Events)"]
    end

    subgraph SimulationSubsystem["OBD-II Simulator Engine (app/simulator/)"]
        MGR["Dataset Manager (services/dataset_manager.py)
        - Column Normalization & Renaming
        - Missing Value Forward-Fill
        - Multi-format loader (CSV / XLSX)"]
        
        ENGINE["Playback Engine (services/simulator.py)
        - Asyncio Background Task Loop
        - State Machine (stopped/running/paused/finished)
        - Speed Scaling (0.5x, 1x, 2x, 5x, 10x)
        - In-memory History Ring Buffer (500 ticks)"]
    end

    subgraph MLSubsystem["ML Inference Subsystem (app/ml/)"]
        DRIVER_MOD["Driver Behaviour Module (driver_behaviour.py)
        - Dynamic Window Buffering
        - Feature Engineering (std, mad, range)
        - StandardScaler + KMeans Model"]
        
        HEALTH_MOD["Vehicle Health Classifier (health_classifier.py)
        - 8-PID Telemetry Vector
        - StandardScaler + Random Forest Model
        - Calibrated Probability Distribution"]
    end

    subgraph StorageLayer["Filesystem Storage"]
        DATASETS["Sample Datasets (/datasets/*.csv)"]
        UPLOADS["User Uploaded Datasets (/uploads/*)"]
        MODELS["Trained ML Artifacts (/models/*.joblib, *.pkl)"]
    end

    DASH & WEBAPP & SWAGGER --> ROUTER_SYS & ROUTER_ML & ROUTER_SIM
    DASH & WEBAPP <--> WS_LIVE & WS_LOGS

    ROUTER_SIM --> ENGINE
    ENGINE <--> MGR
    MGR <--> DATASETS & UPLOADS
    ENGINE --> WS_LIVE & WS_LOGS

    ROUTER_ML --> DRIVER_MOD & HEALTH_MOD
    DRIVER_MOD & HEALTH_MOD <--> MODELS
```

---

## 3. The OBD-II Simulator Subsystem

The simulator emulates an Electronic Control Unit streaming OBD-II data over a Controller Area Network (CAN) bus.

### 3.1 Dataset Ingestion & Schema Normalization
Automotive logging tools (e.g., Torque Pro, OBD Fusion, OpenOBD) generate arbitrary column header formats. The `DatasetManager` (`app/simulator/services/dataset_manager.py`) contains a canonical header dictionary mapping variations into standardized field names:

```
"Engine RPM [RPM]" / "Engine RPM" / "RPM"            ==> rpm
"Vehicle Speed Sensor [km/h]" / "Speed" / "VSS"     ==> vss
"Absolute Throttle Position [%]" / "Throttle"       ==> throttle_pos
"Engine Coolant Temperature [°C]" / "Coolant Temp"  ==> coolant_temp
"Intake Manifold Absolute Pressure [kPa]" / "MAP"   ==> map_kpa
"Mass Air Flow Rate [g/s]" / "MAF"                  ==> maf
"Intake Air Temperature [°C]" / "IAT"               ==> intake_air_temp
"Ambient Air Temperature [°C]"                      ==> ambient_temp
"Accelerator Pedal Position D [%]"                  ==> pedal_d
"Accelerator Pedal Position E [%]"                  ==> pedal_e
```

#### Data Cleaning Protocol:
1. **Header Normalization**: Columns are matched case-insensitively and stripped of special characters.
2. **Missing Value Imputation**: Forward-fill (`ffill`), followed by backward-fill (`bfill`), followed by sensible sensor defaults (e.g., ambient temperature = 25°C, speed = 0 km/h).
3. **Metadata Profiling**: Generates total row counts, duration estimates (assuming standard 10Hz or 1Hz OBD logging rates), and missing-value diagnostics.

### 3.2 Playback Engine & Asyncio Event Loop
The `OBD2Simulator` (`app/simulator/services/simulator.py`) manages a background `asyncio.Task`:

```mermaid
stateDiagram-v2
    [*] --> Stopped: Initial Server Startup

    Stopped --> Running: POST /api/start
    Running --> Paused: POST /api/pause
    Paused --> Running: POST /api/resume
    Running --> Stopped: POST /api/stop or /api/reset
    Paused --> Stopped: POST /api/stop or /api/reset

    Running --> Finished: Last Row Reached (loop=false)
    Running --> Running: Last Row Reached (loop=true, rewind to row 0)
    Finished --> Running: POST /api/start
```

#### Precise Timing Control
- Native OBD logs typically record at $\Delta t = 100\text{ms}$ (10 Hz) or $1000\text{ms}$ (1 Hz).
- The simulator dynamically adjusts sleep intervals based on the configured speed multiplier:
  $$\Delta t_{\text{sleep}} = \frac{\Delta t_{\text{sample}}}{\text{speed\_multiplier}}$$
- Supported multipliers: $0.5\times$, $1.0\times$, $2.0\times$, $5.0\times$, $10.0\times$.

### 3.3 Event Distribution Architecture
Every tick produces a structured event that is broadcast to:
1. **WebSocket Subscribers (`/api/ws/live`)**: Active dashboard sessions receive an instant JSON push.
2. **In-Memory Ring Buffer (`/api/history`)**: The last 500 ticks are cached in a deque, enabling new clients to backfill charts upon connecting.
3. **Snapshot Polling (`/api/live-data`)**: Provides the current tick to REST clients.

---

## 4. Machine Learning Inference Subsystem

The ML subsystem provides decoupled, high-performance predictive intelligence.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Dashboard / Telematics Unit
    participant API as FastAPI Router (app/main.py)
    participant DriverML as Driver Behaviour Engine (K-Means)
    participant HealthML as Vehicle Health Engine (Random Forest)
    participant Disk as Model Store (models/)

    Note over API,Disk: Startup: Deserializes models and standard scalers into memory
    API->>Disk: Load kmeans, rf_health, and standard scalers
    Disk-->>API: Ready for inference

    rect rgb(240, 248, 255)
        Note over Client,DriverML: Workflow 1: Driver Profiling (Window-based)
        Client->>API: POST /api/driver/predict {rpm[], speed[], throttle[]}
        API->>DriverML: predict_from_raw_window(rpm, speed, throttle)
        DriverML->>DriverML: Validate length >= 5
        DriverML->>DriverML: Compute std, mad, accel range (6 features)
        DriverML->>DriverML: Scale with standard_scaler_driver_behavior
        DriverML->>DriverML: KMeans predict centroid index
        DriverML-->>API: {cluster_id, behaviour_class: "Economical"|"Moderate"|"Harsh", features_debug}
        API-->>Client: 200 OK Response
    end

    rect rgb(255, 245, 238)
        Note over Client,HealthML: Workflow 2: Instantaneous Health Classification
        Client->>API: POST /api/health/predict {rpm, throttle, map, maf, coolant, iat, ambient, pedal}
        API->>HealthML: classify_vehicle_health(telemetry_dict)
        HealthML->>HealthML: Validate all 8 features present
        HealthML->>HealthML: Scale with scaler_health
        HealthML->>HealthML: Random Forest predict_proba() across ensemble
        HealthML-->>API: {status: "Normal"|"Warning"|"Critical", confidence: 0.98, probabilities}
        API-->>Client: 200 OK Response
    end
```

### 4.1 Driver Behavior Pipeline
- **Input**: Temporal arrays of `rpm_values`, `speed_values`, and `throttle_values` (length $N \ge 5$).
- **Features Extracted**:
  1. `Engine RPM [RPM]_std`
  2. `Engine RPM [RPM]_mad`
  3. `Vehicle Speed Sensor [km/h]_mad`
  4. `acceleration_std`
  5. `acceleration_range`
  6. `Absolute Throttle Position [%]_std`
- **Output**:
  - Cluster `1` $\to$ **Economical**
  - Cluster `0` $\to$ **Moderate**
  - Cluster `2` $\to$ **Harsh**

For in-depth mathematical derivations, see [K-Means Driver Behavior Model](file:///e:/MP-ECU/ecu-backend/docs/kmeans_driver_behavior.md).

### 4.2 Vehicle Health Pipeline
- **Input**: Instantaneous snapshot containing 8 vital powertrain parameters (`rpm`, `throttle_pos`, `map_kpa`, `maf`, `coolant_temp`, `intake_air_temp`, `ambient_temp`, `pedal_d`).
- **Processing**: Standardizes features and routes them through a pre-trained `RandomForestClassifier` ensemble.
- **Output**:
  - Class prediction (`Normal`, `Warning`, or `Critical`).
  - Prediction confidence (highest class probability).
  - Complete probability distribution across all target classes.

For ensemble mechanics and sensor physical significance, see [Random Forest Vehicle Health Model](file:///e:/MP-ECU/ecu-backend/docs/random_forest_vehicle_health.md).

---

## 5. End-to-End Data Flow: From Raw Sensor to Dashboard UI

The diagram below illustrates how telemetry flows through the entire system during active simulation:

```mermaid
flowchart TD
    A["Raw Dataset File (.csv / .xlsx)"] --> B["DatasetManager.load_and_clean_dataset()"]
    B --> C["Cleaned Pandas DataFrame in Memory"]
    C --> D["OBD2Simulator._playback_loop()"]
    
    subgraph PlaybackLoop["Real-Time Playback Step"]
        D --> E["Extract Row i as Standardized Dict"]
        E --> F["Append to History Deque (max 500)"]
        E --> G["Broadcast to WebSocket Clients (/api/ws/live)"]
        E --> H["Calculate Dynamic Sleep (sample_dt / speed)"]
        H --> D
    end

    G --> I["Client Browser (dashboard.html / Frontend)"]
    
    subgraph ClientProcessing["Client-Side Processing & ML Triggering"]
        I --> J["Render Live Gauges: RPM, Speed, Coolant, Throttle"]
        I --> K["Append to Client Rolling Buffer"]
        
        K -- Every N seconds or ticks --> L["Trigger POST /api/driver/predict
        {rpm_window, speed_window, throttle_window}"]
        
        I -- On each tick or interval --> M["Trigger POST /api/health/predict
        {current_snapshot}"]
    end

    L --> N["Driver Profile Display: Economical / Moderate / Harsh"]
    M --> O["Health Status Indicator: Normal / Warning / Critical"]
```

---

## 6. End-to-End API Route Map

All services are accessible through a clean, RESTful and WebSocket contract:

| Category | Endpoint | Method | Purpose |
| :--- | :--- | :---: | :--- |
| **System** | `/health` | `GET` | Container health probe for zero-downtime platforms |
| **System** | `/` | `GET` | Serves the standalone simulator dashboard |
| **System** | `/docs` | `GET` | Interactive OpenAPI / Swagger interface |
| **System** | `/redoc` | `GET` | ReDoc API specification documentation |
| **ML** | `/api/driver/predict` | `POST` | Ingests rolling window, outputs driver behavior |
| **ML** | `/api/health/predict` | `POST` | Ingests snapshot, outputs powertrain health & confidence |
| **Simulator** | `/api/datasets` | `GET` | Lists available datasets and cleaning metadata |
| **Simulator** | `/api/upload` | `POST` | Uploads and processes new `.csv` / `.xlsx` telemetry |
| **Simulator** | `/api/datasets/{id}` | `DELETE` | Deletes uploaded dataset (bundled sets protected) |
| **Simulator** | `/api/change-dataset` | `POST` | Switches active dataset and rewinds to row 0 |
| **Simulator** | `/api/start` | `POST` | Begins real-time streaming |
| **Simulator** | `/api/pause` | `POST` | Pauses streaming while preserving current row |
| **Simulator** | `/api/resume` | `POST` | Resumes playback from current row |
| **Simulator** | `/api/stop` | `POST` | Terminates playback task |
| **Simulator** | `/api/reset` | `POST` | Stops and rewinds pointer to row 0 |
| **Simulator** | `/api/speed` | `POST` | Adjusts playback speed ($0.5\times$ to $10\times$) |
| **Simulator** | `/api/loop` | `POST` | Sets whether playback loops upon reaching EOF |
| **Simulator** | `/api/status` | `GET` | Returns playback state, current row, elapsed time |
| **Simulator** | `/api/live-data` | `GET` | Polls the latest emitted telemetry tick |
| **Simulator** | `/api/history` | `GET` | Fetches historical ring buffer (default 100 ticks) |
| **Simulator** | `/api/ws/live` | `WS` | Real-time bi-directional telemetry broadcast socket |
| **Simulator** | `/api/ws/logs` | `WS` | Real-time simulator engine log events socket |

---

## 7. Deployment & Hardware Transition Strategy

### Running in Production
The unified architecture requires only one environment configuration:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Or via Docker:
```bash
docker build -t ecu-guardian-backend .
docker run -p 8000:8000 ecu-guardian-backend
```

### Seamless Transition to Physical OBD-II Hardware
When migrating from simulated datasets to real-world vehicle testing:
1. Implement an adapter adhering to the same interface as `OBD2Simulator` in `app/simulator/services/live_adapter.py`.
2. Connect to an ELM327 Bluetooth/USB dongle using `python-OBD`.
3. Query standard PIDs (`0x0C` for RPM, `0x0D` for Speed, `0x05` for Coolant, etc.) and emit the dictionary matching the **Standard Sensor Field Reference**.
4. The rest of the platform — the WebSocket broadcast, dashboard, ML feature engineering, and inference pipelines — requires **zero modifications**.
