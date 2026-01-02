# run_code.py
import os
import argparse
import numpy as np
import torch

from data_utils import prepare_data
from models import MyMMOE
from train_utils import train_and_eval


def set_seed(seed: int = 3407):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train MMOE with one command.")
    parser.add_argument("csv", type=str, help="Input CSV path.")
    parser.add_argument("--id_col", type=str, default=None, help="ID column name (default: first col).")
    parser.add_argument("--label_cols", nargs="*", default=None,
                        help="Label columns (default auto detect: HD PH ... Y).")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--expert_dim", type=int, default=256)
    parser.add_argument("--tower_dims", type=int, nargs=2, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--gradnorm_alpha", type=float, default=0.12)
    parser.add_argument("--gradnorm_lr", type=float, default=0.005)

    parser.add_argument("--outdir", type=str, default="env_outputs")
    parser.add_argument("--save_name", type=str, default="best_mmoe.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    data_pack = prepare_data(
        csv_path=args.csv,
        id_col=args.id_col,
        label_cols=args.label_cols,
        test_size=args.test_size,
        seed=args.seed
    )

    model = MyMMOE(
        input_dim=len(data_pack["used_cols"]),
        task_num=data_pack["task_num"],
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        tower_dims=tuple(args.tower_dims),
        drop=args.dropout
    ).to(device)

    # store kwargs for checkpoint reproducibility
    model.model_kwargs = dict(
        input_dim=len(data_pack["used_cols"]),
        task_num=data_pack["task_num"],
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        tower_dims=tuple(args.tower_dims),
        drop=args.dropout
    )

    train_and_eval(
        model=model,
        data_pack=data_pack,
        device=device,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        save_name=args.save_name
    )


if __name__ == "__main__":
    main()
