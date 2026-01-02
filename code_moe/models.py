# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """A simple residual MLP block used as experts."""
    def __init__(self, dim: int, hidden: int, drop: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(drop)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        h = self.drop(self.act(self.bn1(self.fc1(x))))
        return self.act(x + self.fc2(h))

class MMOE(nn.Module):
    """
    Multi-gate Mixture-of-Experts for multi-task regression.
    Forward returns a list of length task_num, each is [B] tensor.
    """
    def __init__(
        self,
        input_dim: int,
        task_num: int,
        num_experts: int = 6,
        expert_dim: int = 256,
        tower_dims=(256, 128),
        drop: float = 0.25
    ):
        super().__init__()
        self.task_num = task_num
        self.num_experts = num_experts

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.experts = nn.ModuleList([ResidualMLP(expert_dim, expert_dim, drop) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(expert_dim, num_experts) for _ in range(task_num)])

        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, tower_dims[0]), nn.LeakyReLU(inplace=True), nn.Dropout(drop),
                nn.Linear(tower_dims[0], tower_dims[1]), nn.LeakyReLU(inplace=True), nn.Dropout(drop),
                nn.Linear(tower_dims[1], 1)
            ) for _ in range(task_num)
        ])

    def forward(self, x):
        x = self.input_bn(x)
        h = self.shared(x)  # [B, D]

        # expert_outs: [B, D, K]
        expert_outs = torch.stack([E(h) for E in self.experts], dim=2)

        outs = []
        for t in range(self.task_num):
            # w: [B, K]
            w = F.softmax(self.gates[t](h), dim=-1)
            # [B, K, 1]
            w = w.unsqueeze(-1)
            # [B, D, 1] -> [B, D]
            m = torch.bmm(expert_outs, w).squeeze(-1)
            y = self.towers[t](m).squeeze(-1)  # [B]
            outs.append(y)
        return outs
