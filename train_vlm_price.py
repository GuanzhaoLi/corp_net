"""
Train Qwen2-VL (frozen vision) + macro encoder + gated fusion + head for price basis (Route B price).
Usage:
  pip install transformers>=4.45 accelerate
  python train_vlm_price.py --data-dir ./standalone_data --checkpoint-dir ./checkpoints_vlm_price \\
    [--epochs 20] [--batch-size 2] [--normalize-target] [--temporal]

  Year-based validation (no same-year label leakage across train/val):
  python train_vlm_price.py ... --split-by-year [--val-year-fraction 0.2] [--split-seed 42]

  Monthly national macro aligned per subsampled frame (5 frames per sample):
  python train_vlm_price.py ... --monthly-national-macro
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from dataset_vlm_yield import VLMPriceDataset
from model_vlm_yield import build_vlm_price_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_vlm_price")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Small batch: VLM is heavy")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for macro encoder / head / (temporal encoder if --temporal)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (default 0.0)")
    parser.add_argument("--normalize-target", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--yields-csv", default="yields.csv")
    parser.add_argument("--macro-data-csv", default="macro_data.csv")
    parser.add_argument(
        "--macro-features",
        nargs="+",
        default=["crude_oil_usd", "usd_index", "fed_funds_rate", "cpi_yoy", "soybean_corn_ratio"],
    )
    parser.add_argument("--crop", choices=["soybean", "corn"], default="soybean")
    parser.add_argument(
        "--split-by-year",
        action="store_true",
        help="Put entire calendar years in train or val (avoids same price_basis in both splits).",
    )
    parser.add_argument(
        "--val-year-fraction",
        type=float,
        default=0.2,
        help="With --split-by-year: fraction of distinct years assigned to validation (default 0.2).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="With --split-by-year: if set, shuffle years with this seed before taking val years; "
        "if omitted, use chronologically last years as val.",
    )
    parser.add_argument(
        "--monthly-national-macro",
        action="store_true",
        help="Use monthly macro rows aligned to each subsampled satellite frame (B, T=5, M) instead of (B, M).",
    )
    parser.add_argument(
        "--national-monthly-csv",
        default=None,
        help="Optional alternate monthly CSV (date + macro_features). Defaults to {data-dir}/{macro-data-csv}.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    national_csv = (
        os.path.abspath(args.national_monthly_csv)
        if (args.monthly_national_macro and args.national_monthly_csv)
        else None
    )
    dataset = VLMPriceDataset(
        root_dir=args.data_dir,
        yields_csv_name=args.yields_csv,
        macro_data_csv_name=args.macro_data_csv,
        macro_features=args.macro_features,
        crop_type=args.crop,
        use_monthly_national_macro=args.monthly_national_macro,
        national_monthly_csv_path=national_csv,
    )
    if len(dataset) == 0:
        raise SystemExit("No samples. Need yields.csv, macro_data.csv, images/.")

    macro_dim = len(args.macro_features)
    price_mean, price_std = None, None
    if args.normalize_target:
        years = set(s["year"] for s in dataset.samples)
        vals = [dataset.price_basis_lookup[y] for y in years]
        price_mean = float(np.mean(vals))
        price_std = float(np.std(vals)) or 1.0
        print(f"Price basis norm: mean={price_mean:.4f}, std={price_std:.4f}")

    val_years_list = None
    if args.split_by_year:
        year_strs = sorted({str(s["year"]) for s in dataset.samples}, key=lambda y: int(y))
        n_y = len(year_strs)
        if n_y < 2:
            raise SystemExit("--split-by-year needs at least two distinct years in the dataset.")
        n_val = max(1, int(round(n_y * args.val_year_fraction)))
        n_val = min(n_val, n_y - 1)
        if args.split_seed is not None:
            rng = random.Random(args.split_seed)
            shuffled = year_strs[:]
            rng.shuffle(shuffled)
            val_year_set = set(shuffled[:n_val])
        else:
            val_year_set = set(year_strs[-n_val:])
        train_indices = [i for i, s in enumerate(dataset.samples) if s["year"] not in val_year_set]
        val_indices = [i for i, s in enumerate(dataset.samples) if s["year"] in val_year_set]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        val_years_list = sorted(val_year_set, key=int)
        train_years_list = sorted(set(str(s["year"]) for s in dataset.samples) - val_year_set, key=int)
        print(
            f"[split-by-year] train years ({len(train_years_list)}): {train_years_list} | "
            f"val years ({len(val_years_list)}): {val_years_list} | "
            f"samples train={len(train_ds)} val={len(val_ds)}"
        )
    else:
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(
        "Loading Qwen2-VL-2B + macro + gated fusion (vision frozen)"
        + (" + temporal encoder" if args.temporal else "")
        + "..."
    )
    model = build_vlm_price_model(
        macro_input_dim=macro_dim,
        num_frames=5,
        dropout=args.dropout,
        device=device,
        use_temporal=args.temporal,
        monthly_national_macro=args.monthly_national_macro,
    )

    trainable = (
        list(model.price_head.parameters())
        + list(model.macro_encoder.parameters())
        + list(model.gated_fusion.parameters())
    )
    if model.temporal_encoder is not None:
        trainable += list(model.temporal_encoder.parameters())
    if getattr(model, "monthly_national_macro", False):
        trainable += list(model.macro_time_embed.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if price_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "price_basis_norm.json"), "w") as f:
            json.dump({"mean": price_mean, "std": price_std}, f)

    meta = {
        "macro_features": args.macro_features,
        "macro_input_dim": macro_dim,
        "crop": args.crop,
        "use_temporal": args.temporal,
        "split_by_year": bool(args.split_by_year),
        "monthly_national_macro": bool(args.monthly_national_macro),
        "hyperparams": {
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "dropout": float(args.dropout),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "use_temporal": bool(args.temporal),
            "num_frames": 5,
            "normalize_target": bool(args.normalize_target),
            "monthly_national_macro": bool(args.monthly_national_macro),
        },
        "national_monthly_csv": (
            os.path.basename(dataset._national_monthly_csv_resolved)
            if args.monthly_national_macro and dataset._national_monthly_csv_resolved
            else None
        ),
        "national_monthly_csv_abspath": (
            dataset._national_monthly_csv_resolved if args.monthly_national_macro else None
        ),
    }
    if val_years_list is not None:
        meta["val_years"] = val_years_list
        meta["val_year_fraction"] = args.val_year_fraction
        meta["split_seed"] = args.split_seed
    with open(os.path.join(args.checkpoint_dir, "vlm_price_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best_val = float("inf")
    history_train, history_val = [], []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["images"].to(device)
            macro = batch["macro"].to(device)
            targets = batch["price_basis"].to(device).view(-1)
            if price_mean is not None:
                targets = (targets - price_mean) / price_std
            optimizer.zero_grad()
            pred = model(images, macro)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                macro = batch["macro"].to(device)
                targets = batch["price_basis"].to(device).view(-1)
                if price_mean is not None:
                    targets = (targets - price_mean) / price_std
                pred = model(images, macro)
                val_losses.append(criterion(pred, targets).item())
        avg_val = np.mean(val_losses) if val_losses else 0.0
        history_train.append(float(avg_train))
        history_val.append(float(avg_val))
        print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            state = {
                "price_head": model.price_head.state_dict(),
                "macro_encoder": model.macro_encoder.state_dict(),
                "gated_fusion": model.gated_fusion.state_dict(),
                "vision_hidden_size": model.vision_hidden_size,
                "num_frames": model.num_frames,
                "use_temporal": args.temporal,
                "macro_input_dim": macro_dim,
                "monthly_national_macro": bool(args.monthly_national_macro),
            }
            if model.temporal_encoder is not None:
                state["temporal_encoder"] = model.temporal_encoder.state_dict()
            if getattr(model, "monthly_national_macro", False):
                state["macro_time_embed"] = model.macro_time_embed.state_dict()
            torch.save(state, os.path.join(args.checkpoint_dir, "vlm_price_best.pth"))

    state = {
        "price_head": model.price_head.state_dict(),
        "macro_encoder": model.macro_encoder.state_dict(),
        "gated_fusion": model.gated_fusion.state_dict(),
        "vision_hidden_size": model.vision_hidden_size,
        "num_frames": model.num_frames,
        "use_temporal": args.temporal,
        "macro_input_dim": macro_dim,
        "monthly_national_macro": bool(args.monthly_national_macro),
    }
    if model.temporal_encoder is not None:
        state["temporal_encoder"] = model.temporal_encoder.state_dict()
    if getattr(model, "monthly_national_macro", False):
        state["macro_time_embed"] = model.macro_time_embed.state_dict()
    torch.save(state, os.path.join(args.checkpoint_dir, "vlm_price_last.pth"))
    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump({"train_loss": history_train, "val_loss": history_val}, f, indent=2)
    print(f"Saved to {args.checkpoint_dir}. Predict with predict_vlm_price.py.")


if __name__ == "__main__":
    main()
