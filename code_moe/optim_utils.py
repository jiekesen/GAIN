# optim_utils.py
import numpy as np
import torch
import torch.nn.functional as F

class PCGrad(torch.optim.Optimizer):
    """
    PCGrad wrapper for multi-task gradients.
    Use pc_backward(list_of_losses) then step().
    """
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def pc_backward(self, losses):
        grads = []
        shared = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        for L in losses:
            self.zero_grad()
            L.backward(retain_graph=True)
            grads.append([
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in shared
            ])

        for i in range(len(grads)):
            gi = grads[i]
            for j in range(len(grads)):
                if i == j:
                    continue
                gj = grads[j]
                dot = sum((a * b).sum() for a, b in zip(gi, gj))
                if dot < 0:
                    norm = sum((b * b).sum() for b in gj) + 1e-12
                    gi = [a - (dot / norm) * b for a, b in zip(gi, gj)]
            grads[i] = gi

        self.zero_grad()
        for p_idx, p in enumerate(shared):
            p.grad = sum(g[p_idx] for g in grads) / len(grads)

def masked_mse(pred, target, mask):
    """MSE computed only on valid positions (mask==True)."""
    return F.mse_loss(pred[mask], target[mask]) if mask.any() else pred.new_tensor(0.0)

def safe_pearson(y_true, y_pred):
    """Return Pearson correlation; return NaN if variance is zero."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]
