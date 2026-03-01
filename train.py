import argparse
import os

# MPS (Apple Silicon): Transformer with key_padding_mask uses an op not implemented on MPS.
# Enable fallback to CPU for that op so training runs on Mac.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from dataset import CropYieldDataset, AugmentWrapper, augment_temporal_images
from model import CropYieldModel


def collate_crop_yield(batch):
    """Collate samples with variable temporal length T: pad to max_T and return lengths."""
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


def train(epochs_override=None, checkpoint_dir_override=None):
    # 1. Setup
    config = Config()
    if epochs_override is not None:
        config.EPOCHS = epochs_override
    if checkpoint_dir_override is not None:
        config.CHECKPOINT_DIR = checkpoint_dir_override
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data (optionally use yield from CSV for retrain with NASS actual yields)
    yield_csv = getattr(config, "YIELD_CSV", None)
    train_with_nass = getattr(config, "TRAIN_WITH_NASS_CSV", False)
    if train_with_nass and yield_csv:
        fips_codes = config.US_ESTIMATE_FIPS
        years = [str(y) for y in range(2016, 2025)]  # 2016-2024
        print(f"Training with NASS CSV: {len(fips_codes)} FIPS, years {years[0]}-{years[-1]}")
    else:
        fips_codes = config.FIPS_CODES
        years = config.YEARS
    full_dataset = CropYieldDataset(
        root_dir=config.ROOT_DIR,
        years=years,
        fips_codes=fips_codes,
        crop_type=config.CROP_TYPE,
        yield_csv_path=yield_csv,
    )

    # Split 80/20; wrap train with augmentation to reduce constant-output collapse
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset = AugmentWrapper(train_dataset, augment_fn=augment_temporal_images)

    print(f"Data Split: Train={len(train_dataset)}, Val={len(val_dataset)} (train with augmentation)")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_crop_yield
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_crop_yield
    )

    # 3. Model
    model = CropYieldModel(config).to(device)

    # 4. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. Loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    loss_log_path = os.path.join(config.CHECKPOINT_DIR, "loss_log.txt")
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
    with open(loss_log_path, "w") as f:
        f.write("epoch\ttrain_loss\tval_loss\n")

    for epoch in range(config.EPOCHS):
        # Training
        print(f"Epoch [{epoch + 1}/{config.EPOCHS}] Training...")
        model.train()
        batch_train_losses = []
        for batch in train_loader:
            images = batch['images'].to(device)  # (B, T, C, H, W)
            targets = batch['yield'].to(device)  # (B, 1)
            lengths = batch['lengths'].to(device)  # (B,)

            optimizer.zero_grad()
            outputs = model(images, lengths=lengths)
            targets_1d = targets.view(-1)
            loss = criterion(outputs, targets_1d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "GRAD_CLIP", 1.0))
            optimizer.step()
            batch_train_losses.append(loss.item())

        avg_train_loss = np.mean(batch_train_losses)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                targets = batch['yield'].to(device)
                lengths = batch['lengths'].to(device)
                outputs = model(images, lengths=lengths)
                loss = criterion(outputs, targets.view(-1))
                batch_val_losses.append(loss.item())

        avg_val_loss = np.mean(batch_val_losses) if batch_val_losses else 0
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{config.EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch + 1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")

        # Save Best
        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "model_best.pth"))

    # 6. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")

    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "model_last.pth"))
    print("Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override config EPOCHS (e.g. 3 for quick test)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override config CHECKPOINT_DIR")
    args = parser.parse_args()
    train(epochs_override=args.epochs, checkpoint_dir_override=args.checkpoint_dir)
