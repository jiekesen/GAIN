# data_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_LABELS = ["HD", "PH", "PL", "TN", "GP", "SSR", "TGW", "GL", "GW", "Y"]


def infer_id_col(df: pd.DataFrame, id_col: str | None):
    if id_col and id_col in df.columns:
        return id_col
    return df.columns[0]  # fallback: first column


def infer_label_cols(df: pd.DataFrame, label_cols: list[str] | None):
    if label_cols:
        exist = [c for c in label_cols if c in df.columns]
        if len(exist) == 0:
            raise ValueError(f"None of label_cols exist in CSV. You passed: {label_cols}")
        return exist

    exist10 = [c for c in DEFAULT_LABELS if c in df.columns]
    if len(exist10) > 0:
        return exist10

    # fallback single task
    for c in ["hd", "HD", "y", "Y"]:
        if c in df.columns:
            return [c]

    raise ValueError(
        "Cannot infer label columns. Please provide --label_cols explicitly, e.g. --label_cols HD PH ... Y"
    )


def prepare_data(csv_path: str,
                 id_col: str | None,
                 label_cols: list[str] | None,
                 test_size: float,
                 seed: int):
    """
    Read CSV, split train/val, scale X and scale Y per task with masks.
    Returns dict containing everything needed for training.
    """
    df = pd.read_csv(csv_path)

    id_col = infer_id_col(df, id_col)
    label_cols = infer_label_cols(df, label_cols)

    used_cols = [c for c in df.columns if c not in ([id_col] + label_cols)]
    if len(used_cols) == 0:
        raise ValueError("No feature columns found after excluding ID and labels.")

    ids = df[id_col].astype(str).values

    X_raw = df[used_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw)

    Y_raw = df[label_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    mask = ~np.isnan(Y_raw)

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
    Y_tr, Y_te = Y_raw[tr_idx], Y_raw[te_idx]
    M_tr, M_te = mask[tr_idx], mask[te_idx]
    id_tr, id_te = ids[tr_idx], ids[te_idx]

    # Scale X
    x_scaler = StandardScaler()
    X_tr = x_scaler.fit_transform(X_tr).astype(np.float32)
    X_te = x_scaler.transform(X_te).astype(np.float32)
    X_tr = np.nan_to_num(X_tr)
    X_te = np.nan_to_num(X_te)

    # Scale Y per task (only on available labels)
    task_num = len(label_cols)
    y_scalers = []
    Y_tr_z = Y_tr.copy()
    Y_te_z = Y_te.copy()

    for t in range(task_num):
        sc = StandardScaler()
        idx_tr_task = M_tr[:, t]
        if idx_tr_task.sum() >= 2:
            sc.fit(Y_tr[idx_tr_task, t:t+1])
            Y_tr_z[idx_tr_task, t:t+1] = sc.transform(Y_tr[idx_tr_task, t:t+1])
            idx_te_task = M_te[:, t]
            if idx_te_task.sum() > 0:
                Y_te_z[idx_te_task, t:t+1] = sc.transform(Y_te[idx_te_task, t:t+1])
        else:
            class _Id:
                def inverse_transform(self, a): return a
            sc = _Id()
        y_scalers.append(sc)

    Y_tr_z = np.nan_to_num(Y_tr_z)
    Y_te_z = np.nan_to_num(Y_te_z)

    return {
        "df": df,
        "id_col": id_col,
        "label_cols": label_cols,
        "used_cols": used_cols,
        "task_num": len(label_cols),
        "x_scaler": x_scaler,
        "y_scalers": y_scalers,
        "X_tr": X_tr, "X_te": X_te,
        "Y_tr_z": Y_tr_z, "Y_te_z": Y_te_z,
        "M_tr": M_tr, "M_te": M_te,
        "id_tr": id_tr, "id_te": id_te,
    }
