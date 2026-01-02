# data_utils.py
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def detect_id_column(df: pd.DataFrame) -> str:
    """Prefer 'FID' as sample ID, otherwise use the first column."""
    return "FID" if "FID" in df.columns else df.columns[0]

def prepare_train_val(
    csv_path: str,
    label_cols: List[str],
    exclude_cols: List[str],
    test_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    df = pd.read_csv(csv_path)

    # Basic sanity check
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Training CSV missing label columns: {missing_labels}")

    id_col = detect_id_column(df)

    used_cols = [c for c in df.columns if c not in label_cols + exclude_cols]
    if len(used_cols) == 0:
        raise ValueError("No feature columns found after excluding label/exclude columns.")

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df, val_df, used_cols, id_col

def fit_transform_x(train_df: pd.DataFrame, val_df: pd.DataFrame, used_cols: List[str]):
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(train_df[used_cols].values).astype(np.float32)
    X_val = x_scaler.transform(val_df[used_cols].values).astype(np.float32)
    return X_train, X_val, x_scaler

def fit_transform_y(train_df: pd.DataFrame, val_df: pd.DataFrame, label_cols: List[str]):
    """
    Per-task StandardScaler on labels using training valid values only.
    Missing labels are masked; scaled values for missing labels are set to 0.
    """
    Y_train = train_df[label_cols].values.astype(np.float32)
    Y_val = val_df[label_cols].values.astype(np.float32)

    mask_train = ~np.isnan(Y_train)
    mask_val = ~np.isnan(Y_val)

    y_scalers = []
    Y_train_z = Y_train.copy()
    Y_val_z = Y_val.copy()

    for i in range(len(label_cols)):
        scaler = StandardScaler()
        idx = mask_train[:, i]
        if idx.sum() >= 2:
            scaler.fit(Y_train[idx, i:i+1])
            Y_train_z[idx, i:i+1] = scaler.transform(Y_train[idx, i:i+1])

            idx2 = mask_val[:, i]
            if idx2.sum() > 0:
                Y_val_z[idx2, i:i+1] = scaler.transform(Y_val[idx2, i:i+1])
        y_scalers.append(scaler)

    # Fill missing with 0 (ignored by mask in loss)
    Y_train_z[np.isnan(Y_train_z)] = 0
    Y_val_z[np.isnan(Y_val_z)] = 0
    return Y_train_z, Y_val_z, mask_train, mask_val, y_scalers

def save_artifacts(save_dir: str, obj: Dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(obj, f)

def load_artifacts(save_dir: str) -> Dict:
    p = os.path.join(save_dir, "artifacts.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Artifacts not found: {p}. Please run training first.")
    with open(p, "rb") as f:
        return pickle.load(f)
