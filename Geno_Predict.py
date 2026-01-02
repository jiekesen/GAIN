# predict_pipeline.py (FULL, fixed)
import os
import sys
import argparse
import subprocess
import shutil

def run_cmd(cmd, cwd=None):
    print("\n" + "=" * 90)
    print("CWD:", cwd or os.getcwd())
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    parser = argparse.ArgumentParser(description="One-click prediction: VAE encode -> MMOE predict")
    parser.add_argument("input_csv", type=str, help="Input CSV: first col=SampleID, others=VAE raw features")

    parser.add_argument("--vae_dir", type=str, default="code_vae")
    parser.add_argument("--moe_dir", type=str, default="code_moe")

    # ✅ IMPORTANT: default should be 'artifacts' because artifacts.pkl + *.pt are there
    parser.add_argument(
        "--moe_artifacts",
        type=str,
        default="artifacts",
        help="Directory under code_moe containing artifacts.pkl and *.pt (e.g. artifacts)"
    )

    parser.add_argument("--id_col", type=str, default=None, help="Sample ID column name (optional)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="pre_out",
        help="Output directory under current folder (default: pre_out)"
    )
    parser.add_argument(
        "--copy_moe_models",
        action="store_true",
        help="If set, copy artifacts.pkl and *.pt from MOE artifacts dir into pre_out/"
    )
    args = parser.parse_args()

    # ✅ Always use absolute input path (so cwd changes won't break it)
    args.input_csv = os.path.abspath(args.input_csv)

    root = os.path.dirname(os.path.abspath(__file__))
    vae_dir = os.path.join(root, args.vae_dir)
    moe_dir = os.path.join(root, args.moe_dir)

    if not os.path.isdir(vae_dir):
        raise FileNotFoundError(f"VAE dir not found: {vae_dir}")
    if not os.path.isdir(moe_dir):
        raise FileNotFoundError(f"MOE dir not found: {moe_dir}")

    vae_script = os.path.join(vae_dir, "encode_vae.py")
    moe_script = os.path.join(moe_dir, "model_pre.py")
    if not os.path.exists(vae_script):
        raise FileNotFoundError(f"Missing: {vae_script}")
    if not os.path.exists(moe_script):
        raise FileNotFoundError(f"Missing: {moe_script}")

    # ✅ Output directory in current folder
    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ✅ Unified outputs in pre_out/
    vae_feat_out = os.path.join(out_dir, "feat_mu.csv")
    out_pred = os.path.join(out_dir, "predictions.csv")

    # 1) VAE encode -> pre_out/feat_mu.csv
    run_cmd(
        [sys.executable, "encode_vae.py", args.input_csv, "--out", vae_feat_out, "--output_type", "mu"],
        cwd=vae_dir
    )

    # 2) MMOE predict -> pre_out/predictions.csv
    # ✅ Pass absolute save_dir to avoid any cwd confusion
    moe_save_dir_abs = os.path.abspath(os.path.join(moe_dir, args.moe_artifacts))
    if not os.path.exists(os.path.join(moe_save_dir_abs, "artifacts.pkl")):
        raise FileNotFoundError(
            f"artifacts.pkl not found in: {moe_save_dir_abs}\n"
            f"Tip: set --moe_artifacts correctly (usually 'artifacts')."
        )

    cmd = [
        sys.executable, "model_pre.py",
        vae_feat_out,
        "--save_dir", moe_save_dir_abs,
        "--out", out_pred
    ]
    if args.id_col is not None:
        cmd += ["--id_col", args.id_col]

    run_cmd(cmd, cwd=moe_dir)

    # 3) Optional: copy MOE models/artifacts into pre_out for packaging
    if args.copy_moe_models:
        src_dir = moe_save_dir_abs
        for fn in os.listdir(src_dir):
            if fn.endswith(".pt") or fn == "artifacts.pkl" or fn.endswith(".csv"):
                shutil.copy2(os.path.join(src_dir, fn), os.path.join(out_dir, fn))
        print(f"✅ Copied MOE artifacts/models into: {out_dir}")

    print("\n✅ DONE")
    print("Output folder:", out_dir)
    print("VAE feat:", vae_feat_out)
    print("Predictions:", out_pred)

if __name__ == "__main__":
    main()
