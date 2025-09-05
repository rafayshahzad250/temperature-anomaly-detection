import numpy as np
import pandas as pd

def add_rolling_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    out = df.copy()
    out[f'roll_mean_{window}'] = out['temp'].rolling(window).mean()
    out[f'roll_std_{window}'] = out['temp'].rolling(window).std(ddof=0)
    # Z-score wrt rolling mean/std (avoid div by zero)
    eps = 1e-8
    out[f'zscore_{window}'] = (out['temp'] - out[f'roll_mean_{window}']) / (out[f'roll_std_{window}'] + eps)
    return out

def make_windowed_matrix(series: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Build a 2D array of overlapping windows for models like IsolationForest.
    Result shape: [n_windows, window]
    """
    n = len(series)
    if n < window:
        return np.empty((0, window))
    return np.vstack([series[i:i+window] for i in range(n - window + 1)])
