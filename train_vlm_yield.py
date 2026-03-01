"""
Train Route B: Qwen2-VL-2B vision (frozen) + regression head on standalone data.
Usage:
  pip install transformers>=4.45 accelerate
  python train_vlm_yield.py --data-dir ./standalone_data --checkpoint-dir ./checkpoints_vlm [--epochs 20] [--batch-size 2]
"""
import argparse
import os
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset_vlm_yield import VLMYieldDataset
from model_vlm_yield import build_vlm_yield_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_vlm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Small batch: VLM is heavy")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--normalize-target", action="store_true")
    parser.add_argument("--temporal", action="store_true", help="Use 1-layer Transformer over 5 frames to learn temporal pattern (may reduce mean collapse)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = VLMYieldDataset(root_dir=args.data_dir)
    if len(dataset) == 0:
        raise SystemExit("No samples. Check data_dir and yields.csv / images/.")

    yield_mean, yield_std = None, None
    if args.normalize_target:
        all_y = [dataset.yield_lookup[(s["fips"], s["year"])] for s in dataset.samples]
        yield_mean = float(np.mean(all_y))
        yield_std = float(np.std(all_y)) or 1.0
        print(f"Target norm: mean={yield_mean:.2f}, std={yield_std:.2f}")

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading Qwen2-VL-2B + regression head (vision frozen)" + (" + temporal encoder" if args.temporal else "") + "...")
    model = build_vlm_yield_model(num_frames=5, dropout=0.1, device=device, use_temporal=args.temporal)
    trainable = list(model.head.parameters())
    for p in model.head.parameters():
        p.requires_grad = True
    if model.temporal_encoder is not None:
        for p in model.temporal_encoder.parameters():
            p.requires_grad = True
            trainable.append(p)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if yield_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "yield_norm.json"), "w") as f:
            json.dump({"mean": yield_mean, "std": yield_std}, f)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["yield"].to(device).view(-1)
            if yield_mean is not None:
                targets = (targets - yield_mean) / yield_std
            optimizer.zero_grad()
            pred = model(images)
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
                targets = batch["yield"].to(device).view(-1)
                if yield_mean is not None:
                    targets = (targets - yield_mean) / yield_std
                pred = model(images)
                val_losses.append(criterion(pred, targets).item())
        avg_val = np.mean(val_losses) if val_losses else 0.0
        print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            state = {
                "head": model.head.state_dict(),
                "vision_hidden_size": model.vision_hidden_size,
                "num_frames": model.num_frames,
                "use_temporal": args.temporal,
            }
            if model.temporal_encoder is not None:
                state["temporal_encoder"] = model.temporal_encoder.state_dict()
            torch.save(state, os.path.join(args.checkpoint_dir, "vlm_head_best.pth"))

    state = {
        "head": model.head.state_dict(),
        "vision_hidden_size": model.vision_hidden_size,
        "num_frames": model.num_frames,
        "use_temporal": args.temporal,
    }
    if model.temporal_encoder is not None:
        state["temporal_encoder"] = model.temporal_encoder.state_dict()
    torch.save(state, os.path.join(args.checkpoint_dir, "vlm_head_last.pth"))
    print(f"Saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
