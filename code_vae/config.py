# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    feature_start_col: int = 6      # use df.iloc[:, feature_start_col:]
    fillna_value: float = 4.0
    batch_size: int = 128

    # Model
    latent_dim: int = 1024
    encode_dim: List[int] = None
    decode_dim: List[int] = None
    dropout: float = 0.2
    bn: bool = False

    # Loss settings:
    # If binary=True, the decoder uses sigmoid and reconstruction uses BCE.
    # IMPORTANT: BCE assumes input is in [0,1], so we will use MinMax scaling in preprocessing.
    binary: bool = True

    # Training
    lr: float = 1e-4
    var_lr: float = 1e-4
    weight_decay: float = 5e-4
    max_iter: int = 30000
    warmup_n: int = 5000          # KL warm-up steps
    beta: float = 1.0             # max KL weight
    patience: int = 100

    # Output
    outdir: str = "outputs"
    save_model_name: str = "model.pt"
    save_scaler_name: str = "scaler.pkl"
    save_mu_csv: str = "feat_mu.csv"

    # Device
    prefer_cuda: bool = True

    def __post_init__(self):
        if self.encode_dim is None:
            self.encode_dim = [2048, 1024]
        if self.decode_dim is None:
            self.decode_dim = [2048, 1024]
