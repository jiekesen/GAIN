# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class CFG:
    # Reproducibility
    seed: int = 3407

    # Labels (traits)
    label_cols: List[str] = None

    # Columns to exclude from features (besides labels)
    exclude_cols: List[str] = None

    # Train/val split
    test_size: float = 0.2

    # Dataloader
    batch_size_train: int = 256
    batch_size_val: int = 512
    num_workers: int = 4

    # Model
    num_experts: int = 6
    expert_dim: int = 256
    tower_dims: tuple = (256, 128)
    dropout: float = 0.25

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 120
    save_dir: str = "artifacts"
    ckpt_name: str = "best_mmoe.pt"

    # Scheduler
    lr_factor: float = 0.5
    lr_patience: int = 5

    def __post_init__(self):
        if self.label_cols is None:
            self.label_cols = ['HD', 'PH', 'PL', 'TN', 'GP', 'SSR', 'TGW', 'GL', 'GW', 'Y']
        if self.exclude_cols is None:
            self.exclude_cols = ['number']  # keep same as your script
