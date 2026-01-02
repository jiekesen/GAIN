# trainer.py
import os
import numpy as np
import torch
from tqdm import tqdm

class DeterministicWarmup:
    """Linearly increase beta from 0 to t_max over n steps."""
    def __init__(self, n=100, t_max=1.0):
        self.t = 0.0
        self.t_max = float(t_max)
        self.inc = 1.0 / float(n)

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t

class EarlyStopping:
    """
    Early stopping on monitored loss.
    Saves best model state_dict and stops when no improvement for `patience`.
    """
    def __init__(self, patience=20, verbose=False, outdir="outputs", filename="model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf

        os.makedirs(outdir, exist_ok=True)
        self.model_file = os.path.join(outdir, filename)

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
            return

        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
            return

        if score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(loss, model)
        self.counter = 0

    def save_checkpoint(self, loss, model):
        if self.verbose:
            print(f"Loss decreased ({self.loss_min:.6f} -> {loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss

def fit_vae(
    model,
    dataloader,
    lr=1e-4,
    var_lr=1e-4,
    weight_decay=5e-4,
    device="cuda",
    beta=1.0,
    warmup_n=2000,
    max_iter=30000,
    patience=100,
    outdir="outputs",
    ckpt_name="model.pt",
    verbose=False,
):
    """
    Train VAE using KL warm-up and early stopping.
    dataloader yields x batches [B, D].
    """
    model.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.hidden.parameters(), "lr": lr},
            {"params": model.encoder.sample.mu.parameters(), "lr": lr},
            {"params": model.encoder.sample.log_var.parameters(), "lr": var_lr},
            {"params": model.decoder.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    Beta = DeterministicWarmup(n=warmup_n, t_max=beta)
    n_epoch = int(np.ceil(max_iter / len(dataloader)))

    es = EarlyStopping(patience=patience, verbose=verbose, outdir=outdir, filename=ckpt_name)

    loss_hist, rec_hist, kl_hist = [], [], []

    for epoch in tqdm(range(n_epoch), desc="Training"):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_kl = 0.0

        for x in dataloader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.float().to(device)

            optimizer.zero_grad()
            rec_loss, kl_loss = model.compute_loss(x)

            bsz = x.size(0)
            beta_t = next(Beta)
            loss = (rec_loss + beta_t * kl_loss) / bsz

            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered during training.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += (rec_loss.item() / bsz)
            epoch_kl += (kl_loss.item() / bsz)

        loss_hist.append(epoch_loss)
        rec_hist.append(epoch_rec)
        kl_hist.append(epoch_kl)

        es(epoch_loss, model)
        if es.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(es.model_file, map_location=device))

    return {
        "best_ckpt": es.model_file,
        "loss_hist": loss_hist,
        "rec_hist": rec_hist,
        "kl_hist": kl_hist,
    }
