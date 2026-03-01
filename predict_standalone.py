"""
Predict using standalone data layout and (optionally) target-normalized checkpoint.
No cropnet. Reads standalone_data/images/{fips}_{year}.h5 and writes CSV.
Usage:
  python predict_standalone.py --data-dir ./standalone_data --checkpoint ./checkpoints_standalone/model_best.pth --out predictions.csv [--normalize-target]
"""
import argparse
import csv
import os
import json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from config import Config
from dataset_standalone import load_sample_images
from model import CropYieldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data", help="Standalone root (images/ with {fips}_{year}.h5)")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth (or model_last.pth)")
    parser.add_argument("--checkpoint-dir", default=None, help="If set, yield_norm.json is read from here for denormalize")
    parser.add_argument("--normalize-target", action="store_true", help="Denormalize predictions using yield_norm.json from checkpoint-dir")
    parser.add_argument("--out", default="predictions_standalone.csv")
    parser.add_argument("--fips", nargs="+", default=None, help="FIPS to predict (default: all that have images)")
    parser.add_argument("--years", nargs="+", default=None, help="Years (default: all that have images)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    ckpt_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)
    yield_mean, yield_std = None, None
    if args.normalize_target:
        norm_path = os.path.join(ckpt_dir, "yield_norm.json")
        if os.path.isfile(norm_path):
            with open(norm_path) as f:
                n = json.load(f)
            yield_mean = n["mean"]
            yield_std = n["std"]
            print(f"Denormalizing with mean={yield_mean}, std={yield_std}")
        else:
            print("Warning: --normalize-target but yield_norm.json not found; predictions will be in normalized space.")

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CropYieldModel(config).to(device)
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
        with torch.no_grad():
            pred = model(images.to(device), lengths=lengths.to(device))
        y = pred.cpu().float().item()
        if yield_mean is not None and yield_std is not None:
            y = y * yield_std + yield_mean
        rows.append({"fips": fips, "year": year, "predicted_yield_bu_per_acre": y})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "predicted_yield_bu_per_acre"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
