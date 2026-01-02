# src/infer_vae.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import VAEConfig
from preprocessing import load_and_preprocess_csv
from vae.model import VAE

def main():
    cfg = VAEConfig()
    device = cfg.device
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        device = "cpu"

    _, X_scaled, _ = load_and_preprocess_csv(
        cfg.csv_path,
        feature_start_col=cfg.feature_start_col,
        fillna_value=cfg.fillna_value
    )
    X_tensor = torch.from_numpy(X_scaled).float()
    loader = DataLoader(X_tensor, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    input_dim = X_tensor.shape[1]
    dims = [input_dim, cfg.latent_dim, cfg.encode_dim, cfg.decode_dim]
    model = VAE(dims, binary=cfg.binary, dropout=cfg.dropout, bn=cfg.bn)

    ckpt = os.path.join(cfg.outdir, "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    mu = model.encode_batch(loader, device=device, output_type="mu")
    os.makedirs(cfg.outdir, exist_ok=True)
    out_path = os.path.join(cfg.outdir, "mu_features.npy")
    np.save(out_path, mu)

    print("Saved:", out_path, "shape:", mu.shape)

if __name__ == "__main__":
    main()
