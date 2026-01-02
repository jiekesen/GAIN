# src/vae/early_stopping.py
import os
import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping based on validation loss (or training loss if no val loop).
    Saves best model state_dict to disk and reloads on stop.
    """
    def __init__(self, patience=20, verbose=False, outdir=None, filename="model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf

        if outdir is None:
            outdir = "."
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
