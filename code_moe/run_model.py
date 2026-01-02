# run_model.py (FULL, patched: always save + print save paths)
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from config import CFG
from models import MMOE
from optim_utils import PCGrad, masked_mse, safe_pearson
from data_utils import (
    prepare_train_val, fit_transform_x, fit_transform_y,
    save_artifacts
)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MTDataset(Dataset):
    """Dataset returning (X, Y, mask) for multi-task regression."""
    def __init__(self, X, Y, M):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i]

def _ensure_numeric_features(df: pd.DataFrame, cols: list, where: str):
    """
    Force feature columns to numeric and fail fast with clear message if strings exist.
    """
    bad_cols = []
    for c in cols:
        if c not in df.columns:
            bad_cols.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            bad_cols.append(c)

    if bad_cols:
        raise ValueError(
            f"[{where}] Non-numeric or missing feature columns detected (showing up to 30): {bad_cols[:30]}\n"
            f"Tip: Your ID column (e.g., FID/Sample) must NOT be in used_cols. "
            f"Also check if any feature columns are strings."
        )

def main():
    parser = argparse.ArgumentParser(description="Train MMOE with one command.")
    parser.add_argument("train_csv", type=str, help="Training CSV containing labels + features")
    args = parser.parse_args()

    cfg = CFG()
    set_seed(cfg.seed)

    device = torch.device("cuda" if cfg.prefer_cuda and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # -----------------------------
    # Load and split
    # -----------------------------
    train_df, val_df, used_cols, id_col = prepare_train_val(
        args.train_csv, cfg.label_cols, cfg.exclude_cols, cfg.test_size, cfg.seed
    )
    print(f"[INFO] train={len(train_df)} val={len(val_df)}")
    print(f"[INFO] id_col={id_col}")
    print(f"[INFO] #features(before)={len(used_cols)}")

    # -----------------------------
    # SAFETY FIX: Remove ID column from features
    # -----------------------------
    if id_col in used_cols:
        used_cols = [c for c in used_cols if c != id_col]
        print(f"[WARN] Removed id_col from used_cols: {id_col}")

    for maybe_id in ["FID", "Sample", "sample", "ID", "id"]:
        if maybe_id in used_cols and maybe_id != id_col:
            used_cols = [c for c in used_cols if c != maybe_id]
            print(f"[WARN] Removed id-like col from used_cols: {maybe_id}")

    print(f"[INFO] #features(after)={len(used_cols)}")

    _ensure_numeric_features(train_df, used_cols, where="train_df")
    _ensure_numeric_features(val_df, used_cols, where="val_df")

    # -----------------------------
    # Scaling
    # -----------------------------
    X_train, X_val, x_scaler = fit_transform_x(train_df, val_df, used_cols)
    Y_train_z, Y_val_z, mask_train, mask_val, y_scalers = fit_transform_y(
        train_df, val_df, cfg.label_cols
    )

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_loader = DataLoader(
        MTDataset(X_train, Y_train_z, mask_train),
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        MTDataset(X_val, Y_val_z, mask_val),
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # -----------------------------
    # Model + optimizer
    # -----------------------------
    task_num = len(cfg.label_cols)
    model = MMOE(
        input_dim=len(used_cols),
        task_num=task_num,
        num_experts=cfg.num_experts,
        expert_dim=cfg.expert_dim,
        tower_dims=cfg.tower_dims,
        drop=cfg.dropout
    ).to(device)

    base_optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = PCGrad(base_optim)

    log_vars = nn.Parameter(torch.zeros(task_num, device=device))
    base_optim.add_param_group({"params": [log_vars]})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        base_optim, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    # -----------------------------
    # Saving paths (PRINT clearly)
    # -----------------------------
    os.makedirs(cfg.save_dir, exist_ok=True)

    best_ckpt_path = os.path.join(cfg.save_dir, cfg.ckpt_name)       # best
    last_ckpt_path = os.path.join(cfg.save_dir, "last_mmoe.pt")      # always overwrite
    final_ckpt_path = os.path.join(cfg.save_dir, "final_mmoe.pt")    # saved at end

    print(f"[INFO] save_dir   = {cfg.save_dir}")
    print(f"[INFO] best_ckpt  = {best_ckpt_path}")
    print(f"[INFO] last_ckpt  = {last_ckpt_path}")
    print(f"[INFO] final_ckpt = {final_ckpt_path}")

    best_mean_corr = -1.0
    last_task_corr = None

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for Xb, Yb, Mb in train_loader:
            Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
            outs = model(Xb)

            losses = []
            for t in range(task_num):
                L = masked_mse(outs[t], Yb[:, t], Mb[:, t])
                L = torch.exp(-log_vars[t]) * L + log_vars[t]
                losses.append(L)

            optimizer.pc_backward(losses)
            optimizer.step()

            total_loss += torch.stack(losses).mean().item()

        avg_loss = total_loss / max(1, len(train_loader))

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        task_corr = []
        val_mse_list = []

        with torch.no_grad():
            preds, trues = [[] for _ in range(task_num)], [[] for _ in range(task_num)]

            for Xb, Yb, Mb in val_loader:
                Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
                outs = model(Xb)

                for t in range(task_num):
                    mt = Mb[:, t]
                    if mt.any():
                        pz = outs[t][mt].cpu().numpy().reshape(-1, 1)
                        yz = Yb[:, t][mt].cpu().numpy().reshape(-1, 1)

                        p = y_scalers[t].inverse_transform(pz).ravel()
                        y = y_scalers[t].inverse_transform(yz).ravel()

                        preds[t].append(p)
                        trues[t].append(y)

            for t in range(task_num):
                if len(preds[t]) == 0:
                    task_corr.append(np.nan)
                else:
                    p = np.concatenate(preds[t])
                    y = np.concatenate(trues[t])
                    task_corr.append(safe_pearson(y, p))
                    val_mse_list.append(np.mean((p - y) ** 2))

        last_task_corr = task_corr
        mean_corr = float(np.nanmean(task_corr)) if np.any(~np.isnan(task_corr)) else float("nan")
        scheduler.step(float(np.mean(val_mse_list)) if len(val_mse_list) > 0 else 0.0)

        corr_str = ", ".join([
            f"{cfg.label_cols[i]}:{'nan' if np.isnan(c) else f'{c:.3f}'}"
            for i, c in enumerate(task_corr)
        ])
        print(f"[Epoch {epoch:03d}] TrainLoss={avg_loss:.4f} | ValMeanR={mean_corr:.4f} | {corr_str}")

        # ---- ALWAYS save last checkpoint (so you never lose a model) ----
        torch.save(model.state_dict(), last_ckpt_path)

        # ---- Save best checkpoint when improved ----
        if not np.isnan(mean_corr) and mean_corr > best_mean_corr:
            best_mean_corr = mean_corr
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ✅ Improved! Best Mean R={best_mean_corr:.4f} -> saved {best_ckpt_path}")

    # ---- Save final checkpoint once at the end ----
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"✅ Saved final checkpoint -> {final_ckpt_path}")

    # -----------------------------
    # Save preprocessing + metadata
    # -----------------------------
    artifacts = {
        "label_cols": cfg.label_cols,
        "used_cols": used_cols,
        "id_col": id_col,
        "x_scaler": x_scaler,
        "y_scalers": y_scalers,
        "model_kwargs": {
            "input_dim": len(used_cols),
            "task_num": len(cfg.label_cols),
            "num_experts": cfg.num_experts,
            "expert_dim": cfg.expert_dim,
            "tower_dims": cfg.tower_dims,
            "drop": cfg.dropout,
        },
        "seed": cfg.seed,

        # record checkpoints clearly
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "final_ckpt_path": final_ckpt_path,
        "best_mean_corr": best_mean_corr,
    }
    save_artifacts(cfg.save_dir, artifacts)
    print(f"✅ Saved artifacts -> {os.path.join(cfg.save_dir, 'artifacts.pkl')}")

    # save validation summary of the last epoch
    if last_task_corr is None:
        last_task_corr = [np.nan] * len(cfg.label_cols)

    val_summary = pd.DataFrame({
        "Trait": cfg.label_cols,
        "ValPearsonR": last_task_corr
    })
    val_summary_path = os.path.join(cfg.save_dir, "val_summary.csv")
    val_summary.to_csv(val_summary_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved val summary -> {val_summary_path}")

if __name__ == "__main__":
    CFG.prefer_cuda = True
    main()
