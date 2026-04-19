"""
Predict using standalone data layout (must match train_standalone.py).
train_standalone.py trains CropPriceModel (images + macro -> price basis). This script
loads the same architecture and macro features per year.

If standalone_train_meta.json next to the checkpoint sets monthly_national_macro, this script
loads monthly (T, M) macro aligned to satellite dates (same as training). Override with
--force-yearly-macro or --national-monthly-csv.

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
from dataset_standalone import (
    load_sample_images,
    load_macro_vector_for_year,
    build_national_monthly_macro_lookup,
    macro_sequence_for_dates,
)
from model import CropPriceModel


def _resolve_national_monthly_csv(data_dir, ckpt_dir, meta, user_override):
    if user_override:
        p = os.path.abspath(user_override)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"--national-monthly-csv not found: {p}")
    abspath_saved = (meta or {}).get("national_monthly_csv_abspath")
    if abspath_saved and os.path.isfile(abspath_saved):
        return abspath_saved
    name = (meta or {}).get("national_monthly_csv")
    if name:
        base = os.path.basename(name)
        for root in (data_dir, ckpt_dir, os.path.dirname(os.path.abspath(__file__))):
            cand = os.path.join(os.path.abspath(root), base)
            if os.path.isfile(cand):
                return cand
    cand = os.path.join(os.path.abspath(data_dir), "macro_data.csv")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(
        "National monthly macro CSV not found. Pass --national-monthly-csv or place the file in data-dir."
    )


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
    parser.add_argument(
        "--national-monthly-csv",
        default=None,
        help="National monthly macro CSV (for monthly training); auto from standalone_train_meta.json if omitted",
    )
    parser.add_argument(
        "--force-yearly-macro",
        action="store_true",
        help="Use yearly (M,) macro even if checkpoint meta says monthly (will fail if weights expect monthly).",
    )
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

    meta_path = os.path.join(ckpt_dir, "standalone_train_meta.json")
    train_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            train_meta = json.load(f)
    monthly = bool(train_meta.get("monthly_national_macro")) and not args.force_yearly_macro
    config.MONTHLY_NATIONAL_MACRO = monthly

    model = CropPriceModel(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    monthly_lookup = None
    if monthly:
        csv_path = _resolve_national_monthly_csv(args.data_dir, ckpt_dir, train_meta, args.national_monthly_csv)
        monthly_lookup = build_national_monthly_macro_lookup(csv_path, list(args.macro_features))
        print(f"Monthly national macro from {csv_path} (T, M) aligned to satellite dates")

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
            images, dates = load_sample_images(args.data_dir, fips, year, image_subdir="images")
        except FileNotFoundError:
            continue
        images = images.unsqueeze(0)
        lengths = torch.tensor([images.shape[1]], dtype=torch.long)
        if monthly and monthly_lookup is not None:
            arr = macro_sequence_for_dates(
                year,
                dates,
                args.macro_features,
                monthly_lookup,
                len(args.macro_features),
            )
            macro = torch.from_numpy(arr).float().unsqueeze(0)
        else:
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
