# pre.py
import os
import argparse
import numpy as np
import pandas as pd
import torch

from models import MyMMOE


def infer_id_col(df: pd.DataFrame, id_col: str | None):
    """Use provided id_col; otherwise fallback to the first column."""
    if id_col and id_col in df.columns:
        return id_col
    return df.columns[0]


def main():
    parser = argparse.ArgumentParser(description="Load trained MMOE checkpoint and predict on new CSV.")
    parser.add_argument("csv", type=str, help="Input CSV for prediction. First col should be SampleID.")
    parser.add_argument("--ckpt", type=str, default="env_outputs/best_mmoe.pt",
                        help="Path to checkpoint saved by training (default: env_outputs/best_mmoe.pt)")
    parser.add_argument("--id_col", type=str, default=None, help="ID column name (default: first column).")
    parser.add_argument("--outdir", type=str, default="env_pre_out", help="Output folder (default: pre_out)")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output file name (default: predictions.csv)")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n"
                                f"Train first using: python run_code.py train.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load checkpoint (model + scalers + meta)
    # -------------------------
    ckpt = torch.load(ckpt_path, map_location=device)

    used_cols = ckpt["used_cols"]
    label_cols = ckpt["label_cols"]
    saved_id_col = ckpt.get("id_col", None)

    x_scaler = ckpt["x_scaler"]
    y_scalers = ckpt["y_scalers"]

    model_kwargs = ckpt.get("model_kwargs", None)
    if model_kwargs is None:
        # fallback: rebuild using meta
        model_kwargs = {
            "input_dim": len(used_cols),
            "task_num": len(label_cols),
            "num_experts": 6,
            "expert_dim": 256,
            "tower_dims": (256, 128),
            "drop": 0.3
        }

    model = MyMMOE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------
    # Read input CSV
    # -------------------------
    df = pd.read_csv(args.csv)

    # Determine ID col
    if args.id_col is not None:
        id_col = infer_id_col(df, args.id_col)
    else:
        # try to reuse training id_col if exists
        if saved_id_col and saved_id_col in df.columns:
            id_col = saved_id_col
        else:
            id_col = df.columns[0]

    ids = df[id_col].astype(str).values

    # -------------------------
    # Align feature columns
    # -------------------------
    # Case 1: pre.csv already has the same feature names as training (best)
    missing = [c for c in used_cols if c not in df.columns]

    if len(missing) == 0:
        X = df[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    else:
        # Case 2: training used_cols are like ['0','1','2'...] but user's file has no such names
        # Try to interpret features as "all columns except ID and any label columns"
        candidate = df.drop(columns=[id_col], errors="ignore")

        # drop label columns if user mistakenly includes them
        for c in label_cols:
            if c in candidate.columns:
                candidate = candidate.drop(columns=[c])

        # If candidate column count matches training feature count, use by position
        if candidate.shape[1] == len(used_cols):
            X = candidate.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        else:
            # still mismatch -> raise informative error
            raise ValueError(
                f"Test CSV missing feature columns: {missing[:20]} ... (total {len(missing)})\n"
                f"Tip1: Ensure pre.csv contains the SAME feature columns as training.\n"
                f"Tip2: Or make pre.csv contain exactly {len(used_cols)} feature columns (besides ID) "
                f"so it can be matched by position.\n"
                f"Your pre.csv currently has {candidate.shape[1]} feature columns (besides ID)."
            )

    X = np.nan_to_num(X)

    # Scale X using training scaler
    Xz = x_scaler.transform(X).astype(np.float32)
    Xz = np.nan_to_num(Xz)

    # -------------------------
    # Predict
    # -------------------------
    with torch.no_grad():
        xb = torch.tensor(Xz, dtype=torch.float32, device=device)
        outs = model(xb)                            # list length = task_num, each [N]
        preds_z = torch.stack(outs, dim=1).cpu().numpy()  # [N, task_num]

    # Inverse transform each task back to original scale
    preds = preds_z.copy()
    for t in range(len(label_cols)):
        preds[:, t:t+1] = y_scalers[t].inverse_transform(preds[:, t:t+1])

    out_df = pd.DataFrame(preds, columns=label_cols)
    out_df.insert(0, id_col, ids)

    # -------------------------
    # Save outputs
    # -------------------------
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, args.out)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Loaded ckpt: {ckpt_path}")
    print(f"✅ Input: {os.path.abspath(args.csv)} | rows={len(df)}")
    print(f"✅ Saved predictions -> {out_path} | shape={out_df.shape}")


if __name__ == "__main__":
    main()
