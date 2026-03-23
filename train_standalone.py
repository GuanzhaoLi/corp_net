"""
Train on standalone data (no cropnet). Optionally normalize targets to reduce mean collapse.
Usage:
  python train_standalone.py --data-dir ./standalone_data [--normalize-target] [--epochs 30] [--checkpoint-dir ./checkpoints_standalone]
"""
import argparse
import os
import json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

    # macro data
    macro = torch.stack([b["macro"] for b in batch], dim=0)

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
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # initialize the datasets
    full_dataset = StandaloneCropYieldDataset(
        root_dir=args.data_dir,
        yields_csv_name=args.yields_csv,
        macro_data_csv_name=args.macro_data_csv,
        macro_features=args.macro_features,
        image_subdir=args.image_subdir,
        crop_type=args.crop
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

    # assume 80-20 train-val split
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
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE) # Adam optimizer with learning rate from config

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if yield_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "yield_norm.json"), "w") as f:
            json.dump({"mean": yield_mean, "std": yield_std}, f)
    
    if price_basis_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "price_basis_norm.json"), "w") as f:
            json.dump({"mean": price_basis_mean, "std": price_basis_std}, f)

    best_val_loss = float("inf")
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
            outputs = model(images=images, macro=macro, lengths=lengths)
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
                outputs = model(images=images, macro=macro, lengths=lengths)
                val_losses.append(criterion(outputs, target=targets).item())
        avg_val = np.mean(val_losses) if val_losses else 0.0
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_best.pth"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_last.pth"))
    print(f"Saved to {args.checkpoint_dir}. Use predict_standalone.py with --checkpoint-dir and (if used) same --normalize-target.")


if __name__ == "__main__":
    main()
