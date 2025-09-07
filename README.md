# Temperature Anomaly Detection

This repository demonstrates anomaly detection techniques on **simulated temperature sensor data**.  
It covers different anomaly types (spikes, drops, drifts, and noisy/jittery segments) and shows how to combine multiple detectors (rule-based, statistical, and machine learning) into an ensemble.

---

## Project Structure

- **`src/`**  
  Utility code:
  - `sim.py`: synthetic time series generator with injected anomalies.  
  - `features.py`: rolling features and windowing utilities.

- **`notebooks/`**  
  Main workflow, step by step:
  - **01_synthetic_data_and_baselines.ipynb**  
    Generate synthetic temperature data with anomalies. Try simple rule-based baselines.
  - **02_tuning_and_calibration.ipynb**  
    Tune thresholds and hyperparameters for better detection. Explore Isolation Forest.
  - **03_drift_and_jitter_detection.ipynb**  
    Detect slow drifts (rolling slope) and high-variance jitter (rolling std). Build ensemble detectors.
  - **04_streaming_monitor.ipynb**  
    Apply detectors in a streaming loop, simulating real-time monitoring. Saves predictions to `data/processed/`.

- **`data/processed/`**  
  Stores CSV outputs from notebooks (simulated data, predictions, tuned results, etc.).

---

## Setup

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd temperature-anomaly-detection

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
