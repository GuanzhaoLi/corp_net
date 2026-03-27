"""
Predict price basis with trained Qwen2-VL price head (Route B price).
Usage:
  python predict_vlm_price.py --data-dir ./standalone_data \\
    --checkpoint ./checkpoints_vlm_price/vlm_price_best.pth \\
    --out predictions_vlm_price.csv [--normalize-target]
"""
import argparse
import csv
import json
import os

import torch

from dataset_standalone import load_sample_images, load_macro_vector_for_year
from dataset_vlm_yield import DEFAULT_FRAME_INDICES
from model_vlm_yield import build_vlm_price_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-dir", default=None, help="For price_basis_norm.json")
    parser.add_argument("--normalize-target", action="store_true")
    parser.add_argument("--macro-data-csv", default="macro_data.csv")
    parser.add_argument(
        "--macro-features",
        nargs="+",
        default=None,
        help="Must match training; default: read from vlm_price_meta.json next to checkpoint",
    )
    parser.add_argument("--crop", choices=["soybean", "corn"], default=None)
    parser.add_argument("--out", default="predictions_vlm_price.csv")
    parser.add_argument("--fips", nargs="+", default=None)
    parser.add_argument("--years", nargs="+", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)

    meta_path = os.path.join(ckpt_dir, "vlm_price_meta.json")
    macro_features = args.macro_features
    crop = args.crop
    if macro_features is None or crop is None:
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            macro_features = macro_features or meta.get("macro_features")
            crop = crop or meta.get("crop", "soybean")
        if macro_features is None:
            macro_features = [
                "crude_oil_usd",
                "usd_index",
                "fed_funds_rate",
                "cpi_yoy",
                "soybean_corn_ratio",
            ]
        if crop is None:
            crop = "soybean"

    macro_dim = len(macro_features)
    pb_mean, pb_std = None, None
    if args.normalize_target:
        norm_path = os.path.join(ckpt_dir, "price_basis_norm.json")
        if os.path.isfile(norm_path):
            with open(norm_path) as f:
                n = json.load(f)
            pb_mean, pb_std = n["mean"], n["std"]
            print(f"Denormalizing with mean={pb_mean}, std={pb_std}")
        else:
            print("Warning: --normalize-target but price_basis_norm.json missing.")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_temporal = state.get("use_temporal", False)
    if "macro_input_dim" in state:
        assert state["macro_input_dim"] == macro_dim, (
            f"checkpoint macro_input_dim={state['macro_input_dim']} vs CLI {macro_dim}"
        )

    model = build_vlm_price_model(
        macro_input_dim=macro_dim,
        num_frames=len(DEFAULT_FRAME_INDICES),
        device=device,
        use_temporal=use_temporal,
    )
    model.price_head.load_state_dict(state["price_head"], strict=True)
    model.macro_encoder.load_state_dict(state["macro_encoder"], strict=True)
    model.gated_fusion.load_state_dict(state["gated_fusion"], strict=True)
    if model.temporal_encoder is not None and "temporal_encoder" in state:
        model.temporal_encoder.load_state_dict(state["temporal_encoder"], strict=True)
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
        if max(DEFAULT_FRAME_INDICES) >= images.shape[0]:
            continue
        sub = images[DEFAULT_FRAME_INDICES].unsqueeze(0).to(device)
        macro = (
            load_macro_vector_for_year(
                args.data_dir,
                year,
                macro_features,
                macro_data_csv_name=args.macro_data_csv,
                crop_type=crop,
            )
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            y = model(sub, macro).float().item()
        if pb_mean is not None and pb_std is not None:
            y = y * pb_std + pb_mean
        rows.append({"fips": fips, "year": str(year), "predicted_price_basis": y})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "predicted_price_basis"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
