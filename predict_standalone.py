"""
Predict using standalone data layout (must match train_standalone.py).
train_standalone.py trains CropPriceModel (images + macro -> price basis). This script
loads the same architecture and macro features per year.

Usage:
  python predict_standalone.py --data-dir ./standalone_data \\
    --checkpoint ./checkpoints_standalone/model_best.pth --out predictions.csv \\
    [--normalize-target] [--macro-features ...] [--crop soybean]
"""
import argparse
import csv
import os
import json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from config import Config
from dataset_standalone import load_sample_images, load_macro_vector_for_year
from model import CropPriceModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data", help="Standalone root (images/ with {fips}_{year}.h5)")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth (or model_last.pth)")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for price_basis_norm.json (denormalize)")
    parser.add_argument("--normalize-target", action="store_true", help="Denormalize price basis using price_basis_norm.json from checkpoint-dir")
    parser.add_argument("--macro-data-csv", default="macro_data.csv", help="CSV name under data-dir (same as training)")
    parser.add_argument(
        "--macro-features",
        nargs="+",
        default=["crude_oil_usd", "usd_index", "fed_funds_rate", "cpi_yoy", "soybean_corn_ratio"],
        help="Macro columns (must match training)",
    )
    parser.add_argument("--crop", choices=["soybean", "corn"], default="soybean", help="Crop type (must match training)")
    parser.add_argument("--out", default="predictions_standalone.csv")
    parser.add_argument("--fips", nargs="+", default=None, help="FIPS to predict (default: all that have images)")
    parser.add_argument("--years", nargs="+", default=None, help="Years (default: all that have images)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    ckpt_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)
    pb_mean, pb_std = None, None
    if args.normalize_target:
        norm_path = os.path.join(ckpt_dir, "price_basis_norm.json")
        if os.path.isfile(norm_path):
            with open(norm_path) as f:
                n = json.load(f)
            pb_mean = n["mean"]
            pb_std = n["std"]
            print(f"Denormalizing price basis with mean={pb_mean}, std={pb_std}")
        else:
            print(
                "Warning: --normalize-target but price_basis_norm.json not found; "
                "predictions stay in normalized price-basis space (train with --normalize-target to emit the JSON)."
            )

    config = Config()
    if len(args.macro_features) != config.MACRO_INPUT_DIM:
        config.MACRO_INPUT_DIM = len(args.macro_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CropPriceModel(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    base = os.path.join(args.data_dir, "images")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No images dir: {base}")
    if args.fips and args.years:
        pairs = [(f, y) for f in args.fips for y in args.years]
    else:
        pairs = []
        for name in os.listdir(base):
            if name.endswith(".h5") and "_" in name:
                part = name[:-3]
                fips, year = part.split("_", 1)
                pairs.append((fips, year))

    rows = []
    for fips, year in pairs:
        try:
            images, _ = load_sample_images(args.data_dir, fips, year, image_subdir="images")
        except FileNotFoundError:
            continue
        images = images.unsqueeze(0)
        lengths = torch.tensor([images.shape[1]], dtype=torch.long)
        macro = load_macro_vector_for_year(
            args.data_dir,
            year,
            args.macro_features,
            macro_data_csv_name=args.macro_data_csv,
            crop_type=args.crop,
        ).unsqueeze(0)
        with torch.no_grad():
            pred = model(
                images=images.to(device),
                macro_data=macro.to(device),
                lengths=lengths.to(device),
            )
        y = pred.cpu().float().item()
        if pb_mean is not None and pb_std is not None:
            y = y * pb_std + pb_mean
        rows.append({"fips": fips, "year": year, "predicted_price_basis": y})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "predicted_price_basis"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
