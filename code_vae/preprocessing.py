# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_features_from_csv(csv_path: str, feature_start_col: int, fillna_value: float):
    """
    Load CSV and slice features from feature_start_col to end.
    Also return sample names from the first column (optional).
    """
    df = pd.read_csv(csv_path)
    sample_names = df.iloc[:, 0].astype(str).values if df.shape[1] > 0 else None

    X = df.iloc[:, feature_start_col:].copy()
    # Coerce to numeric; non-numeric becomes NaN then filled
    X = X.apply(pd.to_numeric, errors="coerce").fillna(fillna_value)
    X = X.to_numpy(dtype=np.float32)

    return df, sample_names, X

def fit_transform_scaler(X: np.ndarray, binary: bool):
    """
    If binary=True (BCE + sigmoid), map inputs to [0,1] using MinMaxScaler.
    Else use StandardScaler (z-score).
    """
    if binary:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return X_scaled, scaler
