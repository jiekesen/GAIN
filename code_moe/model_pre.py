# code_moe/model_pre.py
import os
import argparse
import numpy as np
import pandas as pd
import torch

from models import MMOE
from data_utils import load_artifacts, detect_id_column


def pick_ckpt(save_dir: str) -> str:
    """Pick checkpoint file in priority order."""
    cand = [
        os.path.join(save_dir, "best_mmoe.pt"),
        os.path.join(save_dir, "last_mmoe.pt"),
        os.path.join(save_dir, "final_mmoe.pt"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p

    # fallback: any .pt
    pt_files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint (.pt) found in {save_dir}")
    return os.path.join(save_dir, sorted(pt_files)[0])


def main():
    parser = argparse.ArgumentParser(description="Load trained MMOE and predict on feature CSV.")
    parser.add_argument("test_csv", type=str, help="CSV containing features (feat_mu). First col is sample ID.")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory containing artifacts.pkl and ckpt")
    parser.add_argument("--out", type=str, default="outputs/predictions.csv", help="Output prediction CSV")
    parser.add_argument("--id_col", type=str, default=None, help="ID column name (optional). Auto-detect if not set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- load artifacts --------
    artifacts = load_artifacts(args.save_dir)

    label_cols = artifacts["label_cols"]
    used_cols = artifacts["used_cols"]
    x_scaler = artifacts["x_scaler"]
    y_scalers = artifacts["y_scalers"]
    model_kwargs = artifacts["model_kwargs"]
    id_col_saved = artifacts.get("id_col", None)

    ckpt_path = artifacts.get("ckpt_path", None)
    if ckpt_path is None or (isinstance(ckpt_path, str) and not os.path.exists(ckpt_path)):
        ckpt_path = pick_ckpt(args.save_dir)

    print(f"[INFO] Using checkpoint: {ckpt_path}")

    # -------- load feature csv --------
    df = pd.read_csv(args.test_csv)

    # Determine ID column
    if args.id_col and args.id_col in df.columns:
        id_col = args.id_col
    elif id_col_saved and id_col_saved in df.columns:
        id_col = id_col_saved
    else:
        id_col = detect_id_column(df)

    # Ensure required feature columns exist
    missing = [c for c in used_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Test CSV missing feature columns: {missing[:20]} ... (total {len(missing)})\n"
            f"Tip: you should pass VAE feat_mu.csv (Sample + 1024 columns) to model_pre.py."
        )

    X = df[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Xz = x_scaler.transform(X).astype(np.float32)

    # -------- build model --------
    model = MMOE(**model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # -------- predict --------
    with torch.no_grad():
        xb = torch.tensor(Xz, dtype=torch.float32, device=device)
        outs = model(xb)
        preds_z = torch.stack(outs, dim=1).cpu().numpy()

    preds = preds_z.copy()
    for t in range(len(label_cols)):
        preds[:, t:t+1] = y_scalers[t].inverse_transform(preds[:, t:t+1])

    out_df = pd.DataFrame(preds, columns=label_cols)
    out_df.insert(0, id_col, df[id_col].astype(str).values)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"✅ Saved predictions -> {args.out} | shape={out_df.shape}")


if __name__ == "__main__":
    main()
