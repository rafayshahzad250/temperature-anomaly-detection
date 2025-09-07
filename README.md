# Temperature Anomaly Detection

This repo contains a simple machine learning project for detecting anomalies in simulated temperature sensor data.  
It demonstrates different anomaly types (spikes, drops, drifts, noisy segments) and combines multiple detectors (rule-based, Isolation Forest, drift/jitter detectors) into an ensemble.  

## Setup

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd temperature-anomaly-detection

python -m venv .venv
source .venv/bin/activate   # on Linux/macOS
# .venv\Scripts\activate    # on Windows

pip install -r requirements.txt
