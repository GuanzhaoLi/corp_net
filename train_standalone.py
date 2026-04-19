"""
Train on standalone data (no cropnet). Optionally normalize targets to reduce mean collapse.
Uses CropPriceModel (images + macro -> annual price_basis). County yield in CSV is still annual.

Optional --monthly-national-macro: build (T, M) from the same monthly table as training
(`{data-dir}/macro_data.csv` by default), aligned to each satellite date. Some columns move
little month-to-month; we still inject the full monthly row so the temporal branch can use it.
County yield / annual price_basis labels are unchanged. Note: yield_bu_acre in that CSV is the
US annual figure repeated each month, not a within-year yield trajectory.

Usage:
  python train_standalone.py --data-dir ./standalone_data [--normalize-target] [--epochs 30] [--checkpoint-dir ./checkpoints_standalone]
  python train_standalone.py ... --monthly-national-macro
  # optional override: --national-monthly-csv /path/to/other_monthly.csv
"""
import argparse
import os
import json
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import pandas as pd

from config import Config
from dataset_standalone import StandaloneCropYieldDataset, AugmentWrapperStandalone, augment_temporal_images
from model import CropYieldModel, CropPriceModel


def collate_crop_yield(batch):
    """Same as train.py: pad variable T, return lengths."""
    images_list = [b["images"] for b in batch]
    max_T = max(im.shape[0] for im in images_list)
    C, H, W = images_list[0].shape[1], images_list[0].shape[2], images_list[0].shape[3]
    lengths = torch.tensor([im.shape[0] for im in images_list], dtype=torch.long) # actual time steps per sample before padding; shape: (batch_size,)
    padded = []
    for im in images_list:
        T = im.shape[0]
        if T < max_T:
            pad = torch.zeros(max_T - T, C, H, W, dtype=im.dtype, device=im.device) # pad with zeros for missing time steps since the time series are variable-length; so the resulting shape: (max_T, C, H, W)
            im = torch.cat([im, pad], dim=0)
        padded.append(im)
    images = torch.stack(padded, dim=0) # build batch tensor of shape (batch_size, max_T, C, H, W)
    yield_ = torch.cat([b["yield"] for b in batch], dim=0)
    return {
        "images": images,
        "yield": yield_,
        "lengths": lengths,
        "fips": [b["fips"] for b in batch],
        "year": [b["year"] for b in batch],
        "dates": [b["dates"] for b in batch],
    }


def collate_crop_price(batch):
    """Collate for crop price prediction. Similar to collate_crop_yield but with "price" instead of "yield"."""
    # visual data
    images_list = [b["images"] for b in batch]
    max_T = max(im.shape[0] for im in images_list)
    C, H, W = images_list[0].shape[1], images_list[0].shape[2], images_list[0].shape[3]
    lengths = torch.tensor([im.shape[0] for im in images_list], dtype=torch.long) # actual time steps per sample before padding; shape: (batch_size,)
    padded = []
    for im in images_list:
        T = im.shape[0]
        if T < max_T:
            pad = torch.zeros(max_T - T, C, H, W, dtype=im.dtype, device=im.device) # pad with zeros for missing time steps since the time series are variable-length; so the resulting shape: (max_T, C, H, W)
            im = torch.cat([im, pad], dim=0)
        padded.append(im)
    images = torch.stack(padded, dim=0) # build batch tensor of shape (batch_size, max_T, C, H, W)

    # macro: (M,) yearly or (T, M) monthly aligned to frames — pad monthly to max_T like images
    macro_list = [b["macro"] for b in batch]
    if macro_list[0].dim() == 1:
        macro = torch.stack(macro_list, dim=0)
    else:
        M = macro_list[0].shape[1]
        macro_pad = []
        for m in macro_list:
            Tm = m.shape[0]
            if Tm < max_T:
                pad = torch.zeros(max_T - Tm, M, dtype=m.dtype, device=m.device)
                m = torch.cat([m, pad], dim=0)
            macro_pad.append(m)
        macro = torch.stack(macro_pad, dim=0)

    # target crop price basis
    price_basis = torch.cat([b["price_basis"] for b in batch], dim=0)

    return {
        "images": images,
        "macro": macro,
        "price_basis": price_basis,
        "lengths": lengths,
        "fips": [b["fips"] for b in batch],
        "year": [b["year"] for b in batch],
        "dates": [b["dates"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data", help="Standalone data root (yields.csv + macro_data.csv + images/)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_standalone", help="Where to save model and yield norm")
    parser.add_argument("--epochs", type=int, default=None, help="Override config EPOCHS")
    parser.add_argument("--normalize-target", action="store_true", help="Z-score normalize yield; save mean/std for prediction")
    parser.add_argument("--yields-csv", default="yields.csv", help="CSV name under data-dir")
    parser.add_argument("--macro-data-csv", default="macro_data.csv", help="CSV name under data-dir")
    parser.add_argument("--macro-features", nargs="+", default=["crude_oil_usd", "usd_index", "fed_funds_rate", "cpi_yoy", "soybean_corn_ratio"], help="Subset of macro features to use. E.g. --macro-features crude_oil_usd usd_index fed_funds_rate cpi_yoy soybean_corn_ratio")
    parser.add_argument("--crop", choices=["soybean", "corn"], help="Crop to model (default: soybean)", default="soybean")
    parser.add_argument("--image-subdir", default="images", help="Subdir under data-dir for H5/npy")
    parser.add_argument(
        "--monthly-national-macro",
        action="store_true",
        help="Use monthly macro rows from data-dir (default: same file as --macro-data-csv, usually macro_data.csv) aligned per frame (T, M) instead of one yearly pooled (M,) vector.",
    )
    parser.add_argument(
        "--national-monthly-csv",
        default=None,
        help="Optional alternate monthly CSV (date + macro_features). If omitted with --monthly-national-macro, uses {data-dir}/{macro-data-csv} e.g. standalone_data/macro_data.csv.",
    )
    # Hyperparameters (override Config defaults) — exposed for grid search
    parser.add_argument("--lr", type=float, default=None, help="Override config.LEARNING_RATE")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.BATCH_SIZE")
    parser.add_argument("--dropout", type=float, default=None, help="Override config.DROPOUT (affects temporal encoder, macro encoder, head)")
    parser.add_argument("--temporal-layers", type=int, default=None, help="Override config.TEMPORAL_LAYERS")
    parser.add_argument("--temporal-heads", type=int, default=None, help="Override config.TEMPORAL_HEADS")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (default 0.0)")
    parser.add_argument(
        "--split-by-year",
        action="store_true",
        help="Put entire calendar years in train or val (avoids same price_basis target in both splits).",
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
        "if omitted, use chronologically last years as val. Also seeds torch/numpy/random for reproducibility.",
    )
    args = parser.parse_args()

    config = Config()
    config.MONTHLY_NATIONAL_MACRO = bool(args.monthly_national_macro)
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.dropout is not None:
        config.DROPOUT = args.dropout
    if args.temporal_layers is not None:
        config.TEMPORAL_LAYERS = args.temporal_layers
    if args.temporal_heads is not None:
        config.TEMPORAL_HEADS = args.temporal_heads

    # Seed for reproducibility when a split seed is provided (so the random_split path is also deterministic)
    if args.split_seed is not None:
        random.seed(args.split_seed)
        np.random.seed(args.split_seed)
        torch.manual_seed(args.split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.split_seed)

    if len(args.macro_features) != config.MACRO_INPUT_DIM:
        config.MACRO_INPUT_DIM = len(args.macro_features)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # initialize the datasets
    national_csv = None
    if args.monthly_national_macro:
        national_csv = args.national_monthly_csv
        if national_csv is not None:
            national_csv = os.path.abspath(national_csv)

    full_dataset = StandaloneCropYieldDataset(
        root_dir=args.data_dir,
        yields_csv_name=args.yields_csv,
        macro_data_csv_name=args.macro_data_csv,
        macro_features=args.macro_features,
        image_subdir=args.image_subdir,
        crop_type=args.crop,
        use_monthly_national_macro=args.monthly_national_macro,
        national_monthly_csv_path=national_csv,
    )
    if len(full_dataset) == 0:
        raise SystemExit("No samples in standalone dataset. Check data_dir and yields.csv / macro_data.csv / images/.")

    # Optionally compute yield mean/std from full dataset (train+val) for normalization
    # TODO: also normalize macro features?
    yield_mean, yield_std = None, None
    price_basis_mean, price_basis_std = None, None
    if args.normalize_target:
        # look up yield for each fips/year in full_dataset.samples, then compute mean/std across all yield for all fips even if there are multiple fips, to get a single global mean/std for normalization
        # this is done on the full dataset (train+val) to avoid data leakage from val to train, since normalization is a global stat that would be computed before train/val split in a real setting
        # if we compute mean/std only on train, then the val set may have different distribution and the normalization may be less effective.
        all_y = [full_dataset.yield_lookup[(s["fips"], s["year"])] for s in full_dataset.samples]
        yield_mean = float(np.mean(all_y))
        yield_std = float(np.std(all_y))
        if yield_std < 1e-6:
            yield_std = 1.0
        
        # look up price basis for each year in full_dataset.samples, then compute mean/std across all price basis for all years to get a single global mean/std for normalization of the price basis prediction target
        all_price_basis = [full_dataset.price_basis_lookup[y] for y in set([s["year"] for s in full_dataset.samples])]
        price_basis_mean = float(np.mean(all_price_basis))
        price_basis_std = float(np.std(all_price_basis))
        if price_basis_std < 1e-6:
            price_basis_std = 1.0
        print(f"Target normalization: yield mean={yield_mean:.2f}, yield std={yield_std:.2f}, price basis mean={price_basis_mean:.2f}, price basis std={price_basis_std:.2f}")

    # Split: either by-year (clean target separation) or random 80/20
    val_years_list = None
    train_years_list = None
    if args.split_by_year:
        year_strs = sorted({str(s["year"]) for s in full_dataset.samples}, key=lambda y: int(y))
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
        train_indices = [i for i, s in enumerate(full_dataset.samples) if s["year"] not in val_year_set]
        val_indices = [i for i, s in enumerate(full_dataset.samples) if s["year"] in val_year_set]
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        val_years_list = sorted(val_year_set, key=int)
        train_years_list = sorted(set(year_strs) - val_year_set, key=int)
        print(
            f"[split-by-year] train years ({len(train_years_list)}): {train_years_list} | "
            f"val years ({len(val_years_list)}): {val_years_list} | "
            f"samples train={len(train_dataset)} val={len(val_dataset)}"
        )
    else:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset = AugmentWrapperStandalone(train_dataset, augment_fn=augment_temporal_images)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_crop_price
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_crop_price
    )

    model = CropPriceModel(config).to(device)
    criterion = nn.MSELoss() # loss function: mean squared error between predicted yield and actual yield
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_meta = {
        "monthly_national_macro": bool(args.monthly_national_macro),
        "national_monthly_csv": (
            os.path.basename(getattr(full_dataset, "_national_monthly_csv_resolved", "") or "")
            if args.monthly_national_macro
            else None
        ),
        "national_monthly_csv_abspath": (
            getattr(full_dataset, "_national_monthly_csv_resolved", None) if args.monthly_national_macro else None
        ),
        "macro_features": list(args.macro_features),
        "crop": args.crop,
        "macro_data_csv": args.macro_data_csv,
        # Hyperparameters actually used (post-override)
        "hyperparams": {
            "lr": float(config.LEARNING_RATE),
            "batch_size": int(config.BATCH_SIZE),
            "dropout": float(config.DROPOUT),
            "temporal_layers": int(config.TEMPORAL_LAYERS),
            "temporal_heads": int(config.TEMPORAL_HEADS),
            "epochs": int(config.EPOCHS),
            "weight_decay": float(args.weight_decay),
            "visual_backbone": str(config.VISUAL_BACKBONE),
            "hidden_dim": int(config.HIDDEN_DIM),
            "macro_input_dim": int(config.MACRO_INPUT_DIM),
            "normalize_target": bool(args.normalize_target),
        },
        "split_by_year": bool(args.split_by_year),
        "val_year_fraction": float(args.val_year_fraction) if args.split_by_year else None,
        "split_seed": args.split_seed,
        "val_years": val_years_list,
        "train_years": train_years_list,
    }
    with open(os.path.join(args.checkpoint_dir, "standalone_train_meta.json"), "w") as f:
        json.dump(train_meta, f, indent=2)
    if yield_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "yield_norm.json"), "w") as f:
            json.dump({"mean": yield_mean, "std": yield_std}, f)
    
    if price_basis_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "price_basis_norm.json"), "w") as f:
            json.dump({"mean": price_basis_mean, "std": price_basis_std}, f)

    best_val_loss = float("inf")
    history_train, history_val = [], []
    for epoch in range(config.EPOCHS):
        model.train()
        batch_losses = []
        for batch in train_loader:
            images = batch["images"].to(device)
            macro = batch["macro"].to(device)
            targets = batch["price_basis"].to(device).view(-1)
            lengths = batch["lengths"].to(device)
            if price_basis_mean is not None:
                targets = (targets - price_basis_mean) / price_basis_std
            optimizer.zero_grad()
            outputs = model(images=images, macro_data=macro, lengths=lengths)
            loss = criterion(outputs, target=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "GRAD_CLIP", 1.0))
            optimizer.step()
            batch_losses.append(loss.item())
        avg_train = np.mean(batch_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                macro = batch["macro"].to(device)
                targets = batch["price_basis"].to(device).view(-1)
                lengths = batch["lengths"].to(device)
                if price_basis_mean is not None:
                    targets = (targets - price_basis_mean) / price_basis_std
                outputs = model(images=images, macro_data=macro, lengths=lengths)
                val_losses.append(criterion(outputs, target=targets).item())
        avg_val = np.mean(val_losses) if val_losses else 0.0
        history_train.append(float(avg_train))
        history_val.append(float(avg_val))
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_best.pth"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_last.pth"))
    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump({"train_loss": history_train, "val_loss": history_val}, f, indent=2)
    print(f"Saved to {args.checkpoint_dir}. Use predict_standalone.py with --checkpoint-dir and (if used) same --normalize-target.")


if __name__ == "__main__":
    main()
