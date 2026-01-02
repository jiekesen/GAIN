# code_vae/encode_vae.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from config import Config
from seed import set_seed
from preprocessing import load_features_from_csv
from model import VAE

def resolve_device(prefer_cuda: bool) -> str:
    if prefer_cuda and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return "cuda"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Encode new samples using a trained VAE (NO training).")
    parser.add_argument("input_csv", type=str, help="Input CSV (first col = sample id, rest = VAE features)")
    parser.add_argument("--ckpt", type=str, default=None, help="VAE checkpoint path. Default: <outdir>/<save_model_name>")
    parser.add_argument("--scaler", type=str, default=None, help="Scaler path. Default: <outdir>/<save_scaler_name>")
    parser.add_argument("--out", type=str, default=None, help="Output feat csv path. Default: <outdir>/feat_mu.csv")
    parser.add_argument("--output_type", type=str, default="mu", choices=["mu", "z", "log_var", "x"])
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = resolve_device(cfg.prefer_cuda)
    print(f"[INFO] device={device}")

    os.makedirs(cfg.outdir, exist_ok=True)

    ckpt_path = args.ckpt or os.path.join(cfg.outdir, cfg.save_model_name)
    scaler_path = args.scaler or os.path.join(cfg.outdir, cfg.save_scaler_name)
    out_csv = args.out or os.path.join(cfg.outdir, cfg.save_mu_csv)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VAE ckpt not found: {ckpt_path} (train VAE first)")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path} (train VAE first)")

    # Load data
    _, sample_names, X = load_features_from_csv(
        args.input_csv,
        feature_start_col=cfg.feature_start_col,
        fillna_value=cfg.fillna_value
    )
    print(f"[INFO] X shape(raw)={X.shape}")

    # Load scaler and transform
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)
    print(f"[INFO] X shape(scaled)={X_scaled.shape}")

    # Build model and load weights
    input_dim = X_scaled.shape[1]
    dims = [input_dim, cfg.latent_dim, cfg.encode_dim, cfg.decode_dim]
    model = VAE(dims, bn=cfg.bn, dropout=cfg.dropout, binary=cfg.binary).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Encode
    X_tensor = torch.from_numpy(X_scaled).float()
    loader = DataLoader(X_tensor, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    feat = model.encode_batch(loader, device=device, output_type=args.output_type)
    print(f"[INFO] feat_{args.output_type} shape={feat.shape}")

    # Save
    out_df = pd.DataFrame(feat)
    if sample_names is not None:
        out_df.insert(0, "Sample", sample_names)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved VAE features -> {out_csv}")

if __name__ == "__main__":
    main()
