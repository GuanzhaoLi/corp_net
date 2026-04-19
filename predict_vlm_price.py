"""
Predict price basis with trained Qwen2-VL price head (Route B price).
Usage:
  python predict_vlm_price.py --data-dir ./standalone_data \\
    --checkpoint ./checkpoints_vlm_price/vlm_price_best.pth \\
    --out predictions_vlm_price.csv [--normalize-target]

  Only (fips, year) pairs that appear in VLMPriceDataset (same as training eligibility):
  python predict_vlm_price.py ... --training-samples-only
"""
import argparse
import csv
import json
import os

import torch

from dataset_standalone import (
    load_sample_images,
    load_macro_vector_for_year,
    build_national_monthly_macro_lookup,
    macro_sequence_for_dates,
)
from dataset_vlm_yield import DEFAULT_FRAME_INDICES, VLMPriceDataset
from model_vlm_yield import build_vlm_price_model


def _resolve_national_monthly_csv(data_dir, ckpt_dir, meta, user_override, macro_data_csv):
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
    cand = os.path.join(os.path.abspath(data_dir), macro_data_csv)
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(
        "National monthly macro CSV not found. Pass --national-monthly-csv or place the file in data-dir."
    )


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
    parser.add_argument(
        "--training-samples-only",
        action="store_true",
        help="Predict only for (fips, year) in VLMPriceDataset (yields + macro + valid price_basis + enough frames).",
    )
    parser.add_argument(
        "--yields-csv",
        default="yields.csv",
        help="With --training-samples-only: must match training (default yields.csv).",
    )
    parser.add_argument(
        "--national-monthly-csv",
        default=None,
        help="Monthly macro CSV for monthly-trained checkpoints. Auto-detected from meta if omitted.",
    )
    parser.add_argument(
        "--force-yearly-macro",
        action="store_true",
        help="Use (B, M) yearly macro even if checkpoint meta says monthly.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)

    meta_path = os.path.join(ckpt_dir, "vlm_price_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    macro_features = args.macro_features or meta.get("macro_features")
    crop = args.crop or meta.get("crop")
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
    monthly = bool(state.get("monthly_national_macro", meta.get("monthly_national_macro", False)))
    if args.force_yearly_macro:
        monthly = False
    if "macro_input_dim" in state:
        assert state["macro_input_dim"] == macro_dim, (
            f"checkpoint macro_input_dim={state['macro_input_dim']} vs CLI {macro_dim}"
        )

    model = build_vlm_price_model(
        macro_input_dim=macro_dim,
        num_frames=len(DEFAULT_FRAME_INDICES),
        device=device,
        use_temporal=use_temporal,
        monthly_national_macro=monthly,
    )
    model.price_head.load_state_dict(state["price_head"], strict=True)
    model.macro_encoder.load_state_dict(state["macro_encoder"], strict=True)
    model.gated_fusion.load_state_dict(state["gated_fusion"], strict=True)
    if model.temporal_encoder is not None and "temporal_encoder" in state:
        model.temporal_encoder.load_state_dict(state["temporal_encoder"], strict=True)
    if monthly and "macro_time_embed" in state:
        model.macro_time_embed.load_state_dict(state["macro_time_embed"], strict=True)
    model.eval()

    monthly_lookup = None
    if monthly:
        csv_path = _resolve_national_monthly_csv(
            args.data_dir, ckpt_dir, meta, args.national_monthly_csv, args.macro_data_csv
        )
        monthly_lookup = build_national_monthly_macro_lookup(csv_path, list(macro_features))
        print(f"Monthly national macro from {csv_path} aligned per subsampled frame")

    base = os.path.join(args.data_dir, "images")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No images dir: {base}")

    if args.training_samples_only:
        vds = VLMPriceDataset(
            root_dir=args.data_dir,
            yields_csv_name=args.yields_csv,
            macro_data_csv_name=args.macro_data_csv,
            macro_features=macro_features,
            crop_type=crop,
        )
        pairs = [(s["fips"], s["year"]) for s in vds.samples]
        print(f"[training-samples-only] {len(pairs)} pairs from VLMPriceDataset")
        if args.fips and args.years:
            want = {(str(f).zfill(5), str(int(y))) for f in args.fips for y in args.years}
            pairs = [(f, y) for f, y in pairs if (str(f).zfill(5), str(int(y))) in want]
            print(f"  after --fips/--years filter: {len(pairs)} pairs")
    elif args.fips and args.years:
        pairs = [(str(f).zfill(5), str(int(y))) for f in args.fips for y in args.years]
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
        if max(DEFAULT_FRAME_INDICES) >= images.shape[0]:
            continue
        sub = images[DEFAULT_FRAME_INDICES].unsqueeze(0).to(device)
        if monthly and monthly_lookup is not None:
            sub_dates = [dates[i] for i in DEFAULT_FRAME_INDICES]
            arr = macro_sequence_for_dates(
                year, sub_dates, macro_features, monthly_lookup, len(macro_features)
            )
            macro = torch.from_numpy(arr).float().unsqueeze(0).to(device)
        else:
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
