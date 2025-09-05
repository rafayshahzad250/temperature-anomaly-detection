from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_temperature_series(
    n: int = 10_000,
    base_temp: float = 100.0,
    noise_sigma: float = 0.2,
    seed: int | None = 42,
    anomaly_specs: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Simulate a temperature time series with optional anomalies.
    Returns a DataFrame with columns: ['t', 'temp', 'is_anomaly', 'anomaly_type']

    anomaly_specs: list of dicts, each like:
      {'kind': 'spike', 'idx': 3000, 'magnitude': 20.0}
      {'kind': 'drop', 'idx': 4500, 'magnitude': -15.0}
      {'kind': 'drift', 'start': 6000, 'end': 8000, 'slope': 0.005}
      {'kind': 'noisy', 'start': 2000, 'end': 2200, 'sigma': 1.0}
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    temp = base_temp + rng.normal(0, noise_sigma, size=n)

    is_anomaly = np.zeros(n, dtype=bool)
    anomaly_type = np.array(["none"] * n, dtype=object)

    if anomaly_specs is None:
        anomaly_specs = [
            {'kind': 'spike', 'idx': int(n * 0.3), 'magnitude': 20.0},
            {'kind': 'drop', 'idx': int(n * 0.55), 'magnitude': -15.0},
            {'kind': 'drift', 'start': int(n * 0.7), 'end': int(n * 0.9), 'slope': 0.01},
            {'kind': 'noisy', 'start': int(n * 0.15), 'end': int(n * 0.18), 'sigma': 1.2},
        ]

    for spec in anomaly_specs:
        kind = spec['kind']
        if kind in ('spike', 'drop'):
            idx = spec['idx']
            temp[idx] += spec['magnitude']
            is_anomaly[idx] = True
            anomaly_type[idx] = kind
        elif kind == 'drift':
            start, end, slope = spec['start'], spec['end'], spec['slope']
            drift = slope * np.arange(end - start)
            temp[start:end] += drift
            is_anomaly[start:end] = True
            anomaly_type[start:end] = 'drift'
        elif kind == 'noisy':
            start, end, sigma = spec['start'], spec['end'], spec['sigma']
            temp[start:end] += rng.normal(0, sigma, size=end - start)
            is_anomaly[start:end] = True
            anomaly_type[start:end] = 'noisy'
        else:
            raise ValueError(f"Unknown anomaly kind: {kind}")

    return pd.DataFrame({
        't': t,
        'temp': temp,
        'is_anomaly': is_anomaly,
        'anomaly_type': anomaly_type
    })
