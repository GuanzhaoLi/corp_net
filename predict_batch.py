"""
Batch prediction: predict soybean yield for multiple FIPS from Sentinel data only (no USDA).
Use when data lives on AWS or elsewhere: pass --root-dir to the Sentinel data root.
Output: CSV with fips, year, predicted_yield_bu_per_acre. Feed to aggregate_to_us.py for US estimate.
"""
import argparse
import csv
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from config import Config
from dataset import load_sentinel_for_prediction, _state_abbr_from_fips
from model import CropYieldModel


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict county soybean yield from Sentinel H5 (no USDA). Output CSV for aggregate_to_us.py."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root dir of Sentinel data (e.g. on AWS: /data or s3 mount). Structure: root_dir/Sentinel/data/AG/{year}/{state_abbr}/*.h5",
    )
    parser.add_argument(
        "--sentinel-subpath",
        type=str,
        default=None,
        help='Subpath under root-dir to AG data. Default: "Sentinel/data/AG". For AG_chen layout use "AG".',
    )
    parser.add_argument(
        "--fips",
        type=str,
        nargs="+",
        default=None,
        help="FIPS codes to predict (default: config.US_ESTIMATE_FIPS, 15 representative counties)",
    )
    parser.add_argument("--year", type=str, help="Single year (e.g. 2024)")
    parser.add_argument(
        "--years",
        type=str,
        nargs="+",
        default=None,
        help="Multiple years (e.g. 2016 2017 ... 2025). Overrides --year.",
    )
    parser.add_argument("--out", type=str, default="batch_predictions.csv", help="Output CSV path")
    parser.add_argument("--debug", action="store_true", help="Print path, n_timesteps, image mean/std per (fips, year)")
    args = parser.parse_args()

    years = args.years if args.years else ([args.year] if args.year else None)
    if not years:
        raise SystemExit("Provide --year or --years (e.g. --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025)")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = Config()
    fips_list = args.fips or config.US_ESTIMATE_FIPS
    print(f"Predicting {len(fips_list)} FIPS × {len(years)} years")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = CropYieldModel(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    rows = []
    for year in years:
        for fips in fips_list:
            try:
                images, dates = load_sentinel_for_prediction(
                    args.root_dir, fips, year, sentinel_ag_subpath=args.sentinel_subpath
                )
            except Exception as e:
                print(f"Skip {fips} {year}: {e}")
                continue
            if args.debug:
                subpath = args.sentinel_subpath or os.path.join("Sentinel", "data", "AG")
                data_path = os.path.join(args.root_dir, subpath, str(year), _state_abbr_from_fips(fips))
                n_t = images.shape[0]
                mean_ = images.float().mean().item()
                std_ = images.float().std().item()
                print(f"  [debug] {fips} {year} path={data_path} n_timesteps={n_t} mean={mean_:.4f} std={std_:.4f}")
            images = images.unsqueeze(0)
            lengths = torch.tensor([images.shape[1]], dtype=torch.long)
            with torch.no_grad():
                pred = model(images.to(device), lengths=lengths.to(device))
            yield_bu = pred.cpu().float().item()
            rows.append({"fips": fips, "year": year, "predicted_yield_bu_per_acre": yield_bu})
            print(f"  {fips} {year}: {yield_bu:.2f} bu/acre")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "predicted_yield_bu_per_acre"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
