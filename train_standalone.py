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

from config import Config
from dataset_standalone import StandaloneCropYieldDataset, AugmentWrapperStandalone, augment_temporal_images
from model import CropYieldModel


def collate_crop_yield(batch):
    """Same as train.py: pad variable T, return lengths."""
    images_list = [b["images"] for b in batch]
    max_T = max(im.shape[0] for im in images_list)
    C, H, W = images_list[0].shape[1], images_list[0].shape[2], images_list[0].shape[3]
    lengths = torch.tensor([im.shape[0] for im in images_list], dtype=torch.long)
    padded = []
    for im in images_list:
        T = im.shape[0]
        if T < max_T:
            pad = torch.zeros(max_T - T, C, H, W, dtype=im.dtype, device=im.device)
            im = torch.cat([im, pad], dim=0)
        padded.append(im)
    images = torch.stack(padded, dim=0)
    yield_ = torch.cat([b["yield"] for b in batch], dim=0)
    return {
        "images": images,
        "yield": yield_,
        "lengths": lengths,
        "fips": [b["fips"] for b in batch],
        "year": [b["year"] for b in batch],
        "dates": [b["dates"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data", help="Standalone data root (yields.csv + images/)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_standalone", help="Where to save model and yield norm")
    parser.add_argument("--epochs", type=int, default=None, help="Override config EPOCHS")
    parser.add_argument("--normalize-target", action="store_true", help="Z-score normalize yield; save mean/std for prediction")
    parser.add_argument("--yields-csv", default="yields.csv", help="CSV name under data-dir")
    parser.add_argument("--image-subdir", default="images", help="Subdir under data-dir for H5/npy")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    full_dataset = StandaloneCropYieldDataset(
        root_dir=args.data_dir,
        yields_csv_name=args.yields_csv,
        image_subdir=args.image_subdir,
    )
    if len(full_dataset) == 0:
        raise SystemExit("No samples in standalone dataset. Check data_dir and yields.csv / images/.")

    # Optionally compute yield mean/std from full dataset (train+val) for normalization
    yield_mean, yield_std = None, None
    if args.normalize_target:
        all_y = [full_dataset.yield_lookup[(s["fips"], s["year"])] for s in full_dataset.samples]
        yield_mean = float(np.mean(all_y))
        yield_std = float(np.std(all_y))
        if yield_std < 1e-6:
            yield_std = 1.0
        print(f"Target normalization: mean={yield_mean:.2f}, std={yield_std:.2f}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset = AugmentWrapperStandalone(train_dataset, augment_fn=augment_temporal_images)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_crop_yield
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_crop_yield
    )

    model = CropYieldModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if yield_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "yield_norm.json"), "w") as f:
            json.dump({"mean": yield_mean, "std": yield_std}, f)

    best_val_loss = float("inf")
    for epoch in range(config.EPOCHS):
        model.train()
        batch_losses = []
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["yield"].to(device).view(-1)
            lengths = batch["lengths"].to(device)
            if yield_mean is not None:
                targets = (targets - yield_mean) / yield_std
            optimizer.zero_grad()
            outputs = model(images, lengths=lengths)
            loss = criterion(outputs, targets)
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
                targets = batch["yield"].to(device).view(-1)
                lengths = batch["lengths"].to(device)
                if yield_mean is not None:
                    targets = (targets - yield_mean) / yield_std
                outputs = model(images, lengths=lengths)
                val_losses.append(criterion(outputs, targets).item())
        avg_val = np.mean(val_losses) if val_losses else 0.0
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_best.pth"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_last.pth"))
    print(f"Saved to {args.checkpoint_dir}. Use predict_standalone.py with --checkpoint-dir and (if used) same --normalize-target.")


if __name__ == "__main__":
    main()
