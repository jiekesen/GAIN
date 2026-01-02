# model_pre.py
# ------------------------------------------------------------
# Load trained MMOE + preprocessing artifacts and predict on new CSV.
# Single-folder friendly: run anywhere with `python model_pre.py xxx.csv`.
# Output will always contain a sample id column and NO unwanted index column.
# ------------------------------------------------------------

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Make sure single-folder imports work no matter where you run from
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import MMOE
from data_utils import load_artifacts, detect_id_column


def drop_possible_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop common accidental index columns introduced by CSV saving/loading.

    IMPORTANT:
    - Do NOT drop '0','1','2',... because those can be REAL feature columns
      (e.g., VAE latent features saved as 0..1023).
    """
    drop_set = {"Unnamed: 0", "Unnamed: 0.1", "index", "Index"}
    drop_cols = [c for c in df.columns if str(c).strip() in drop_set]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def choose_sample_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    """
    Decide which column to use as sample id/name.

    Priority:
    1) preferred (saved id_col) if exists
    2) 'FID' if exists
    3) 'Sample' if exists
    4) first column
    """
    if preferred and preferred in df.columns:
        return preferred
    if "FID" in df.columns:
        return "FID"
    if "Sample" in df.columns:
        return "Sample"
    return df.columns[0] if df.shape[1] > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Load trained MMOE and predict on new CSV.")
    parser.add_argument("test_csv", type=str, help="CSV containing features (same columns as training)")
    parser.add_argument("--save_dir", type=str, default="artifacts", help="Directory containing artifacts.pkl and ckpt")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output prediction CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(args.save_dir)

    # ---- Load artifacts ----
    label_cols = artifacts["label_cols"]          # list of trait names
    used_cols = artifacts["used_cols"]            # list of feature columns used in training
    id_col_saved = artifacts.get("id_col", None)  # training-time sample id column name
    x_scaler = artifacts["x_scaler"]              # fitted StandardScaler for X
    y_scalers = artifacts["y_scalers"]            # list of per-task scalers for Y
    ckpt_path = artifacts["ckpt_path"]            # path to best model weights
    model_kwargs = artifacts["model_kwargs"]      # model init kwargs

    # ---- Read input CSV ----
    df = pd.read_csv(args.test_csv)
    df = drop_possible_index_columns(df)

    # ---- Choose sample id column ----
    id_col = choose_sample_column(df, id_col_saved)

    if id_col is None:
        # No columns -> create dummy ids
        sample_ids = np.array([f"sample_{i}" for i in range(len(df))], dtype=str)
        id_col_out = "Sample"
    else:
        sample_ids = df[id_col].astype(str).values
        id_col_out = id_col

    # ---- Check required feature columns ----
    missing = [c for c in used_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Test CSV missing feature columns: {missing[:20]} ... (total {len(missing)})\n"
            f"Tip: ensure your test CSV contains the SAME feature columns as training.\n"
            f"Example: if your training used VAE features '0'..'1023', test CSV must also have them."
        )

    # ---- Build X and scale (do NOT fit again) ----
    X = df[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Xz = x_scaler.transform(X).astype(np.float32)

    # ---- Build model and load weights ----
    model = MMOE(**model_kwargs).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first using run_model.py")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---- Predict ----
    with torch.no_grad():
        xb = torch.tensor(Xz, dtype=torch.float32, device=device)
        outs = model(xb)  # list length=task_num, each shape [N]
        preds_z = torch.stack(outs, dim=1).cpu().numpy()  # [N, task_num]

    # ---- Inverse transform predictions to original scale ----
    preds = preds_z.copy()
    for t in range(len(label_cols)):
        preds[:, t:t+1] = y_scalers[t].inverse_transform(preds[:, t:t+1])

    # ---- Save output (NO index column) ----
    out_df = pd.DataFrame(preds, columns=label_cols)
    out_df.insert(0, id_col_out, sample_ids)

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"✅ Saved predictions -> {args.out} | shape={out_df.shape}")


if __name__ == "__main__":
    main()
