# src/train_vae.py
"""
Train a VAE model on tabular features extracted from a CSV file.

Workflow:
1) Load CSV
2) Slice features from a specified start column
3) Fill missing values
4) Standardize features (z-score)
5) Train VAE with KL warm-up + early stopping
6) Save best model checkpoint and the fitted scaler for consistent inference

Outputs (default):
- outputs/model.pt       (best VAE checkpoint)
- outputs/scaler.pkl     (StandardScaler fitted on training data)
"""

import os
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import VAEConfig
from seed import set_seed
from preprocessing import load_and_preprocess_csv
from vae.model import VAE
from trainer import fit_vae


def resolve_device(cfg_device: str) -> str:
    """Resolve device string based on availability."""
    if cfg_device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return "cuda"
    return "cpu"


def main():
    # ----------------------------
    # Load configuration
    # ----------------------------
    cfg = VAEConfig()

    # ----------------------------
    # Reproducibility
    # ----------------------------
    set_seed(cfg.seed)

    # ----------------------------
    # Device
    # ----------------------------
    device = resolve_device(cfg.device)
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # Output directory
    # ----------------------------
    outdir = cfg.outdir or "outputs"
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------
    # Load & preprocess data
    # ----------------------------
    # data_df: original full dataframe (unused in training, but returned)
    # X_scaled: standardized features (float32)
    # scaler: fitted StandardScaler (saved for inference)
    data_df, X_scaled, scaler = load_and_preprocess_csv(
        csv_path=cfg.csv_path,
        feature_start_col=cfg.feature_start_col,
        fillna_value=cfg.fillna_value,
    )

    print(f"[INFO] Loaded data: {cfg.csv_path}")
    print(f"[INFO] Total rows: {len(data_df)}")
    print(f"[INFO] Feature matrix shape: {X_scaled.shape} (from col {cfg.feature_start_col} to end)")

    # Save scaler for consistent inference
    scaler_path = os.path.join(outdir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler: {scaler_path}")

    # ----------------------------
    # Build DataLoader
    # ----------------------------
    X_tensor = torch.from_numpy(X_scaled).float()

    train_loader = DataLoader(
        X_tensor,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,    # consistent batch size helps stability
        num_workers=0,     # set >0 if you want faster loading
        pin_memory=(device == "cuda"),
    )

    # ----------------------------
    # Build model
    # ----------------------------
    input_dim = X_tensor.shape[1]
    dims = [input_dim, cfg.latent_dim, cfg.encode_dim, cfg.decode_dim]
    model = VAE(dims=dims, bn=cfg.bn, dropout=cfg.dropout, binary=cfg.binary)

    print("[INFO] VAE initialized")
    print(f"[INFO] input_dim={input_dim}, latent_dim={cfg.latent_dim}, binary={cfg.binary}")
    print(f"[INFO] encode_dim={cfg.encode_dim}, decode_dim={cfg.decode_dim}")

    # ----------------------------
    # Train
    # ----------------------------
    stats = fit_vae(
        model=model,
        dataloader=train_loader,
        lr=cfg.lr,
        var_lr=cfg.var_lr,
        weight_decay=cfg.weight_decay,
        device=device,
        beta=cfg.beta,
        warmup_n=cfg.warmup_n,
        max_iter=cfg.max_iter,
        patience=cfg.patience,
        outdir=outdir,
        verbose=False,
    )

    print("[INFO] Training finished.")
    print("[INFO] Best checkpoint:", stats["best_ckpt"])
    print("[INFO] Final epochs:", len(stats["loss"]))

    # Optionally save training curves as .npy for later plotting
    np.save(os.path.join(outdir, "loss_hist.npy"), np.array(stats["loss"], dtype=np.float32))
    np.save(os.path.join(outdir, "recon_hist.npy"), np.array(stats["recon"], dtype=np.float32))
    np.save(os.path.join(outdir, "kl_hist.npy"), np.array(stats["kl"], dtype=np.float32))
    print("[INFO] Saved training curves: loss_hist.npy / recon_hist.npy / kl_hist.npy")


if __name__ == "__main__":
    main()
