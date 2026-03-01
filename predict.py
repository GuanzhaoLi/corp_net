"""
Predict crop yield using a trained model.
Loads Sentinel-2 imagery for given FIPS/year, runs the model, and writes predictions to CSV.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import CropYieldDataset
from model import CropYieldModel
from train import collate_crop_yield


def main():
    parser = argparse.ArgumentParser(description="Predict crop yield from Sentinel-2 imagery")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: config.CHECKPOINT_DIR/model_best.pth)",
    )
    parser.add_argument(
        "--fips",
        type=str,
        nargs="+",
        default=None,
        help="FIPS codes to predict (default: config.FIPS_CODES)",
    )
    parser.add_argument(
        "--years",
        type=str,
        nargs="+",
        default=None,
        help="Years to predict (default: config.YEARS)",
    )
    parser.add_argument(
        "--crop-type",
        type=str,
        default=None,
        help="Crop type (default: config.CROP_TYPE)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Data root (default: config.ROOT_DIR)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    args = parser.parse_args()

    config = Config()
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    root_dir = args.root_dir or config.ROOT_DIR
    fips_codes = args.fips or config.FIPS_CODES
    years = args.years or config.YEARS
    crop_type = args.crop_type or config.CROP_TYPE

    # Build dataset (only samples that have both Sentinel and yield data will be included)
    dataset = CropYieldDataset(
        root_dir=root_dir,
        years=years,
        fips_codes=fips_codes,
        crop_type=crop_type,
    )
    if len(dataset) == 0:
        print("No samples to predict. Check root_dir, fips, years, and that Sentinel + USDA data exist.")
        return

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collate_crop_yield,
    )

    # Load model
    model = CropYieldModel(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            lengths = batch["lengths"].to(device)
            pred = model(images, lengths=lengths)
            pred_np = pred.cpu().float().numpy()
            gt = batch["yield"].cpu().float().numpy()
            for i in range(len(batch["fips"])):
                row = {
                    "fips": batch["fips"][i],
                    "year": batch["year"][i],
                    "predicted_yield_bu_per_acre": float(pred_np[i]),
                    "actual_yield_bu_per_acre": float(gt[i, 0]),
                }
                rows.append(row)

    # Write CSV
    import csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fips", "year", "predicted_yield_bu_per_acre", "actual_yield_bu_per_acre"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions written to {args.out} ({len(rows)} rows)")

    # Print summary
    if rows:
        preds = [r["predicted_yield_bu_per_acre"] for r in rows]
        actuals = [r["actual_yield_bu_per_acre"] for r in rows]
        mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(rows)
        print(f"RMSE (on this set): {mse ** 0.5:.4f} bu/acre")


if __name__ == "__main__":
    main()
