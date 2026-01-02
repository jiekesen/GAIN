# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(drop)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        h = self.act(self.bn1(self.fc1(x)))
        h = self.drop(h)
        return self.act(x + self.fc2(h))


class MyMMOE(nn.Module):
    """
    MMOE backbone:
      - input BN
      - shared projection
      - K experts (ResidualMLP)
      - T gates (softmax over experts)
      - T towers (regression heads)
    """
    def __init__(self, input_dim, task_num, num_experts=6, expert_dim=256, tower_dims=(256, 128), drop=0.3):
        super().__init__()
        self.task_num = task_num
        self.num_experts = num_experts

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.LeakyReLU()
        )

        self.experts = nn.ModuleList([ResidualMLP(expert_dim, expert_dim, drop) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(expert_dim, num_experts) for _ in range(task_num)])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, tower_dims[0]), nn.LeakyReLU(), nn.Dropout(drop),
                nn.Linear(tower_dims[0], tower_dims[1]), nn.LeakyReLU(), nn.Dropout(drop),
                nn.Linear(tower_dims[1], 1)
            ) for _ in range(task_num)
        ])

    def forward(self, x):
        x = torch.nan_to_num(x)
        x = self.input_bn(x)
        h = self.shared(x)

        # expert_outs: [B, D, K]
        expert_outs = torch.stack([E(h) for E in self.experts], dim=2)

        outs = []
        for t in range(self.task_num):
            gate = F.softmax(self.gates[t](h), dim=-1).unsqueeze(-1)  # [B, K, 1]
            mix = torch.bmm(expert_outs, gate).squeeze(-1)            # [B, D]
            outs.append(self.towers[t](mix).squeeze(-1))              # [B]
        return outs
