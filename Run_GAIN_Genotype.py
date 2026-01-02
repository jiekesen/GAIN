# Run_GAIN_Genotype.py
# ------------------------------------------------------------
# Stable one-click pipeline:
#   1) Train/encode VAE (code_vae)
#   2) Merge phenotype (label) + VAE features (feat_mu)
#   3) Train MMOE (code_moe)
#
# Usage:
#   python Run_GAIN_Genotype.py --vae_input /abs/path/test.csv --label /abs/path/phe.csv --id_col FID
#
# If phe has NO id column, attach IDs using id_file:
#   python Run_GAIN_Genotype.py --vae_input /abs/path/test.csv --label /abs/path/phe.csv \
#       --id_col FID --id_file /abs/path/id.csv
# ------------------------------------------------------------

import os
import sys
import argparse
import subprocess
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def run_cmd(cmd, cwd=None):
    print("\n" + "=" * 90)
    print("CWD:", cwd or os.getcwd())
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True, cwd=cwd)


def must_exist(path, hint=""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\n{hint}")


def looks_like_a_path(s: str) -> bool:
    return ("/" in s) or ("\\" in s) or s.lower().endswith(".csv")


def detect_id_col(df: pd.DataFrame, preferred: str):
    """Try to detect the ID column name in df."""
    if preferred in df.columns:
        return preferred
    for c in ["FID", "Sample", "sample", "ID", "id", "Name", "name"]:
        if c in df.columns:
            return c
    return df.columns[0]


def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Stable pipeline: VAE -> merge -> MMOE")

    # inputs
    parser.add_argument("--vae_input", required=True, help="CSV for VAE (first col = sample id)")
    parser.add_argument("--label", required=True, help="Phenotype CSV for MMOE training")
    parser.add_argument("--id_col", default="FID", help="ID column NAME to merge on (e.g. FID or Sample)")

    # optional id file (only when phe has no id_col)
    parser.add_argument("--id_file", default=None, help="Optional: CSV containing IDs to attach to phe rows")
    parser.add_argument("--id_file_col", default=None, help="Optional: column name in id_file (default first col)")

    # project folders
    parser.add_argument("--vae_dir", default="code_vae", help="Folder containing VAE code")
    parser.add_argument("--mmoe_dir", default="code_moe", help="Folder containing MMOE code")

    # entry scripts
    parser.add_argument("--vae_entry", default="run_model.py", help="VAE entry script inside code_vae/")
    parser.add_argument("--mmoe_entry", default="run_model.py", help="MMOE entry script inside code_moe/")

    # outputs
    parser.add_argument("--vae_feat_out", default=None,
                        help="Path to VAE feat csv. Default: <code_vae>/outputs/feat_mu.csv")
    parser.add_argument("--merged_out", default=None,
                        help="Path to merged MMOE train csv. Default: <code_moe>/outputs/mmoe_train_merged.csv")

    args = parser.parse_args()

    # Guard: --id_col must be column name, not a file path
    if looks_like_a_path(args.id_col):
        raise ValueError(
            f"--id_col should be a COLUMN NAME (e.g., FID), but you passed: {args.id_col}\n"
            f"If you want to pass an id.csv file, use --id_file /path/to/id.csv"
        )

    # ---- ABSOLUTE PATH FIX (most important) ----
    args.vae_input = to_abs(args.vae_input)
    args.label = to_abs(args.label)
    if args.id_file is not None:
        args.id_file = to_abs(args.id_file)

    # Project root and code dirs
    root = os.path.dirname(os.path.abspath(__file__))
    vae_dir = os.path.join(root, args.vae_dir)
    mmoe_dir = os.path.join(root, args.mmoe_dir)

    must_exist(vae_dir, "Make sure code_vae exists at project root.")
    must_exist(mmoe_dir, "Make sure code_moe exists at project root.")

    vae_entry_path = os.path.join(vae_dir, args.vae_entry)
    mmoe_entry_path = os.path.join(mmoe_dir, args.mmoe_entry)
    must_exist(vae_entry_path, f"VAE entry not found: {vae_entry_path}")
    must_exist(mmoe_entry_path, f"MMOE entry not found: {mmoe_entry_path}")

    must_exist(args.vae_input, "Check --vae_input path.")
    must_exist(args.label, "Check --label path.")
    if args.id_file is not None:
        must_exist(args.id_file, "Check --id_file path.")

    # Default outputs
    if args.vae_feat_out is None:
        args.vae_feat_out = os.path.join(vae_dir, "outputs", "feat_mu.csv")
    if args.merged_out is None:
        args.merged_out = os.path.join(mmoe_dir, "outputs", "mmoe_train_merged.csv")

    os.makedirs(os.path.dirname(args.vae_feat_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.merged_out), exist_ok=True)

    # ------------------------------------------------------------
    # 1) Run VAE (inside code_vae)
    # ------------------------------------------------------------
    # NOTE: pass absolute input path so it won't break when cwd changes
    run_cmd([sys.executable, args.vae_entry, args.vae_input], cwd=vae_dir)

    must_exist(
        args.vae_feat_out,
        f"VAE feature CSV not found: {args.vae_feat_out}\n"
        f"Make sure VAE writes to code_vae/outputs/feat_mu.csv or pass --vae_feat_out explicitly."
    )

    # ------------------------------------------------------------
    # 2) Load VAE features and normalize ID column name
    # ------------------------------------------------------------
    feat_df = pd.read_csv(args.vae_feat_out)
    print(f"[INFO] Loaded VAE features: {args.vae_feat_out} | shape={feat_df.shape}")

    feat_id_col = detect_id_col(feat_df, preferred="Sample")  # VAE usually outputs Sample
    if feat_id_col != args.id_col:
        feat_df = feat_df.rename(columns={feat_id_col: args.id_col})
        print(f"[INFO] Rename VAE feature id col '{feat_id_col}' -> '{args.id_col}'")

    # ------------------------------------------------------------
    # 3) Load phenotype/label and ensure it has id_col
    # ------------------------------------------------------------
    label_df = pd.read_csv(args.label)
    print(f"[INFO] Loaded phe/label: {args.label} | shape={label_df.shape}")

    if args.id_col not in label_df.columns:
        # attach from id_file if provided
        if args.id_file is None:
            raise ValueError(
                f"phe/label CSV has no id column '{args.id_col}'. Columns: {list(label_df.columns)[:30]}\n"
                f"Fix: add ID column to phe.csv OR provide --id_file /path/to/id.csv"
            )

        id_df = pd.read_csv(args.id_file)
        id_src_col = args.id_file_col or id_df.columns[0]
        if id_src_col not in id_df.columns:
            raise ValueError(f"id_file_col='{id_src_col}' not found in id_file columns: {list(id_df.columns)}")

        if len(id_df) != len(label_df):
            raise ValueError(
                f"id_file rows ({len(id_df)}) != phe rows ({len(label_df)}). They must match row-by-row."
            )

        # attach id as the FIRST column
        label_df.insert(0, args.id_col, id_df[id_src_col].astype(str).values)
        print(f"[INFO] Attached id column '{args.id_col}' from id_file='{args.id_file}', col='{id_src_col}'")

    # ------------------------------------------------------------
    # 4) Deduplicate & merge
    # ------------------------------------------------------------
    if feat_df[args.id_col].duplicated().any():
        feat_df = feat_df.drop_duplicates(subset=[args.id_col], keep="first")
        print("[WARN] Duplicated IDs in VAE features. Kept first occurrence.")
    if label_df[args.id_col].duplicated().any():
        label_df = label_df.drop_duplicates(subset=[args.id_col], keep="first")
        print("[WARN] Duplicated IDs in phe/label. Kept first occurrence.")

    merged = pd.merge(label_df, feat_df, on=args.id_col, how="inner")

    if merged.shape[0] == 0:
        raise ValueError(
            "Merge result is empty. IDs between phe/label and VAE features do not match.\n"
            f"Check the values in '{args.id_col}'."
        )

    # Ensure stable column order: ID + phe columns + feature columns
    phe_cols = list(label_df.columns)               # includes id_col
    feat_cols = [c for c in feat_df.columns if c != args.id_col]
    ordered_cols = [args.id_col] + [c for c in phe_cols if c != args.id_col] + feat_cols
    merged = merged[ordered_cols]

    merged.to_csv(args.merged_out, index=False, encoding="utf-8-sig")
    print(f"✅ Merged MMOE training CSV saved -> {args.merged_out} | shape={merged.shape}")

    # ------------------------------------------------------------
    # 5) Run MMOE training (inside code_moe)
    # ------------------------------------------------------------
    # NOTE: pass absolute merged_out to avoid cwd issues
    run_cmd([sys.executable, args.mmoe_entry, to_abs(args.merged_out)], cwd=mmoe_dir)

    print("\n🎉 Done: VAE -> MMOE pipeline finished successfully.")
    print(f"[INFO] VAE output: {args.vae_feat_out}")
    print(f"[INFO] MMOE input: {args.merged_out}")
    print(f"[INFO] MMOE outputs folder: {os.path.join(mmoe_dir, 'outputs')}")


if __name__ == "__main__":
    main()
