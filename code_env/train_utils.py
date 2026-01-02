# train_utils.py
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MTDataset(Dataset):
    def __init__(self, X, Y, M, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.bool)
        self.ids = np.array(ids).astype(str)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i], self.ids[i]


def masked_mse(pred, target, mask):
    if mask.any():
        loss = F.mse_loss(pred[mask], target[mask])
        if torch.isnan(loss):
            return torch.zeros([], requires_grad=True, device=pred.device)
        return loss
    return torch.zeros([], requires_grad=True, device=pred.device)


def safe_pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


class GradNorm:
    """
    GradNorm reweights task losses to balance training dynamics.
    """
    def __init__(self, model, alpha=0.12, lr=0.005):
        self.model = model
        self.alpha = alpha
        self.lr = lr
        self.initial_losses = None

    def update_weights(self, task_losses, task_weights):
        g_param = self.model.shared[0].weight
        grads = []
        for L in task_losses:
            g = torch.autograd.grad(L, g_param, retain_graph=True, allow_unused=True)[0]
            g_norm = torch.tensor(0.0, device=task_weights.device) if g is None else g.norm()
            grads.append(g_norm)

        grads = torch.stack([torch.nan_to_num(g, nan=0.0) for g in grads])
        losses = torch.stack(task_losses).detach()

        if self.initial_losses is None:
            self.initial_losses = losses.clone()

        loss_ratio = torch.clamp(losses / (self.initial_losses + 1e-8), 0.5, 2.0)
        mean_grad = grads.mean()
        targets = mean_grad * (loss_ratio ** self.alpha)

        with torch.no_grad():
            new_w = task_weights - self.lr * (grads - targets)
            new_w = torch.clamp(new_w, 1e-3, 1.0)
            new_w = new_w / new_w.sum()
        return new_w


def train_and_eval(model,
                   data_pack: dict,
                   device: torch.device,
                   outdir: str,
                   epochs: int,
                   batch_size: int,
                   lr: float,
                   weight_decay: float,
                   gradnorm_alpha: float,
                   gradnorm_lr: float,
                   save_name: str):
    os.makedirs(outdir, exist_ok=True)

    label_cols = data_pack["label_cols"]
    task_num = data_pack["task_num"]
    x_scaler = data_pack["x_scaler"]
    y_scalers = data_pack["y_scalers"]
    used_cols = data_pack["used_cols"]
    id_col = data_pack["id_col"]

    train_loader = DataLoader(
        MTDataset(data_pack["X_tr"], data_pack["Y_tr_z"], data_pack["M_tr"], data_pack["id_tr"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        MTDataset(data_pack["X_te"], data_pack["Y_te_z"], data_pack["M_te"], data_pack["id_te"]),
        batch_size=max(batch_size, 512),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_vars = nn.Parameter(torch.zeros(task_num, device=device))
    optimizer.add_param_group({"params": [log_vars]})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    gradnorm = GradNorm(model, alpha=gradnorm_alpha, lr=gradnorm_lr)
    task_weights = torch.ones(task_num, device=device) / task_num

    best_mean_corr = -1.0
    best_path = os.path.join(outdir, save_name)

    start = time.time()
    last_task_corr = [np.nan] * task_num

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for Xb, Yb, Mb, _ in train_loader:
            Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
            outs = model(Xb)

            losses = []
            for t in range(task_num):
                L = masked_mse(outs[t], Yb[:, t], Mb[:, t])
                L = torch.exp(-log_vars[t]) * L + log_vars[t]
                losses.append(L)

            task_weights = gradnorm.update_weights(losses, task_weights)
            batch_loss = sum(task_weights[t] * losses[t] for t in range(task_num))

            if torch.isnan(batch_loss):
                continue

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            total_loss += float(batch_loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step(epoch)

        # ---- validation ----
        model.eval()
        preds_all = [[] for _ in range(task_num)]
        trues_all = [[] for _ in range(task_num)]

        with torch.no_grad():
            for Xb, Yb, Mb, _ids in val_loader:
                Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
                outs = model(Xb)

                for t in range(task_num):
                    mt = Mb[:, t]
                    if mt.any():
                        pz = outs[t][mt].detach().cpu().numpy().reshape(-1, 1)
                        yz = Yb[:, t][mt].detach().cpu().numpy().reshape(-1, 1)
                        preds_all[t].append(pz)
                        trues_all[t].append(yz)

        task_corr = []
        for t in range(task_num):
            if len(preds_all[t]) == 0:
                task_corr.append(np.nan)
                continue
            pz = np.concatenate(preds_all[t], axis=0)
            yz = np.concatenate(trues_all[t], axis=0)
            p = y_scalers[t].inverse_transform(pz).ravel()
            y = y_scalers[t].inverse_transform(yz).ravel()
            task_corr.append(safe_pearson(p, y))

        last_task_corr = task_corr
        mean_corr = float(np.nanmean(task_corr)) if np.any(~np.isnan(task_corr)) else 0.0

        msg = f"[Epoch {epoch:03d}] TrainLoss={avg_loss:.4f} | ValCorr={mean_corr:.4f} | "
        msg += ", ".join([f"{label_cols[i]}:{'nan' if np.isnan(c) else f'{c:.3f}'}"
                          for i, c in enumerate(task_corr)])
        print(msg)

        if mean_corr > best_mean_corr:
            best_mean_corr = mean_corr
            torch.save({
                "model_state": model.state_dict(),
                "x_scaler": x_scaler,
                "y_scalers": y_scalers,
                "used_cols": used_cols,
                "label_cols": label_cols,
                "id_col": id_col,
                "model_kwargs": model.model_kwargs if hasattr(model, "model_kwargs") else None,
            }, best_path)
            print(f"  ✅ Improved! Saved -> {best_path}")

    # ---- export predictions on val set using best model ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_te = data_pack["X_te"]
    id_te = data_pack["id_te"]
    Y_te_z = data_pack["Y_te_z"]
    M_te = data_pack["M_te"]

    X_te_tensor = torch.tensor(X_te, dtype=torch.float32, device=device)
    with torch.no_grad():
        outs = model(X_te_tensor)
        preds_z = torch.stack(outs, dim=1).cpu().numpy()

    preds = preds_z.copy()
    for t in range(task_num):
        preds[:, t:t+1] = y_scalers[t].inverse_transform(preds[:, t:t+1])

    pred_df = pd.DataFrame(preds, columns=label_cols)
    pred_df.insert(0, id_col, id_te)
    pred_path = os.path.join(outdir, "pred_val.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    met_df = pd.DataFrame({"Trait": label_cols, "PearsonR": last_task_corr})
    met_path = os.path.join(outdir, "metrics.csv")
    met_df.to_csv(met_path, index=False, encoding="utf-8-sig")

    minutes = (time.time() - start) / 60.0
    print(f"\n🏁 Done. Best MeanR={best_mean_corr:.4f} | time={minutes:.2f} min")
    print(f"Saved: {best_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {met_path}")

    return best_path
