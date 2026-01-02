# run_model.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from config import Config
from seed import set_seed
from preprocessing import load_features_from_csv, fit_transform_scaler
from model import VAE
from trainer import fit_vae

def resolve_device(prefer_cuda: bool) -> str:
    if prefer_cuda and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return "cuda"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(
        description="Train VAE and export feat_mu with ONE command."
    )
    parser.add_argument("input_csv", type=str, help="Input CSV file path, e.g., test.csv")
    parser.add_argument("--output_type", type=str, default="mu", choices=["mu", "z", "log_var", "x"],
                        help="Which representation to export")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    device = resolve_device(cfg.prefer_cuda)
    print(f"[INFO] device={device}")

    os.makedirs(cfg.outdir, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    _, sample_names, X = load_features_from_csv(
        args.input_csv,
        feature_start_col=cfg.feature_start_col,
        fillna_value=cfg.fillna_value
    )
    print(f"[INFO] input_csv={args.input_csv}")
    print(f"[INFO] X shape (raw) = {X.shape}")

    # -------------------------
    # Scaling (matches binary mode)
    # -------------------------
    X_scaled, scaler = fit_transform_scaler(X, binary=cfg.binary)
    print(f"[INFO] X shape (scaled) = {X_scaled.shape} | binary={cfg.binary}")

    scaler_path = os.path.join(cfg.outdir, cfg.save_scaler_name)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] saved scaler -> {scaler_path}")

    # -------------------------
    # Dataloaders
    # -------------------------
    X_tensor = torch.from_numpy(X_scaled).float()
    train_loader = DataLoader(
        X_tensor,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda"),
        num_workers=0
    )

    infer_loader = DataLoader(
        X_tensor,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
        num_workers=0
    )

    # -------------------------
    # Build VAE model
    # -------------------------
    input_dim = X_tensor.shape[1]
    dims = [input_dim, cfg.latent_dim, cfg.encode_dim, cfg.decode_dim]
    model = VAE(dims, bn=cfg.bn, dropout=cfg.dropout, binary=cfg.binary)

    # -------------------------
    # Train
    # -------------------------
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
        outdir=cfg.outdir,
        ckpt_name=cfg.save_model_name,
        verbose=False,
    )
    print(f"[INFO] best_ckpt={stats['best_ckpt']}")

    # -------------------------
    # Export features
    # -------------------------
    feat = model.encode_batch(infer_loader, device=device, output_type=args.output_type)
    print(f"[INFO] feat_{args.output_type} shape = {feat.shape}")

    out_csv = os.path.join(cfg.outdir, cfg.save_mu_csv)
    out_df = pd.DataFrame(feat)
    if sample_names is not None:
        out_df.insert(0, "Sample", sample_names)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved features -> {out_csv}")

if __name__ == "__main__":
    main()
