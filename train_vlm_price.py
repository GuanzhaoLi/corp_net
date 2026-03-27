"""
Train Qwen2-VL (frozen vision) + macro encoder + gated fusion + head for price basis (Route B price).
Usage:
  pip install transformers>=4.45 accelerate
  python train_vlm_price.py --data-dir ./standalone_data --checkpoint-dir ./checkpoints_vlm_price \\
    [--epochs 20] [--batch-size 2] [--normalize-target] [--temporal]
"""
import argparse
import os
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset_vlm_yield import VLMPriceDataset
from model_vlm_yield import build_vlm_price_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_vlm_price")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Small batch: VLM is heavy")
    parser.add_argument("--lr", type=float, default=1e-3)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = VLMPriceDataset(
        root_dir=args.data_dir,
        yields_csv_name=args.yields_csv,
        macro_data_csv_name=args.macro_data_csv,
        macro_features=args.macro_features,
        crop_type=args.crop,
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
        dropout=0.1,
        device=device,
        use_temporal=args.temporal,
    )

    trainable = (
        list(model.price_head.parameters())
        + list(model.macro_encoder.parameters())
        + list(model.gated_fusion.parameters())
    )
    if model.temporal_encoder is not None:
        trainable += list(model.temporal_encoder.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if price_mean is not None:
        with open(os.path.join(args.checkpoint_dir, "price_basis_norm.json"), "w") as f:
            json.dump({"mean": price_mean, "std": price_std}, f)

    with open(os.path.join(args.checkpoint_dir, "vlm_price_meta.json"), "w") as f:
        json.dump(
            {
                "macro_features": args.macro_features,
                "macro_input_dim": macro_dim,
                "crop": args.crop,
                "use_temporal": args.temporal,
            },
            f,
            indent=2,
        )

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
            }
            if model.temporal_encoder is not None:
                state["temporal_encoder"] = model.temporal_encoder.state_dict()
            torch.save(state, os.path.join(args.checkpoint_dir, "vlm_price_best.pth"))

    state = {
        "price_head": model.price_head.state_dict(),
        "macro_encoder": model.macro_encoder.state_dict(),
        "gated_fusion": model.gated_fusion.state_dict(),
        "vision_hidden_size": model.vision_hidden_size,
        "num_frames": model.num_frames,
        "use_temporal": args.temporal,
        "macro_input_dim": macro_dim,
    }
    if model.temporal_encoder is not None:
        state["temporal_encoder"] = model.temporal_encoder.state_dict()
    torch.save(state, os.path.join(args.checkpoint_dir, "vlm_price_last.pth"))
    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump({"train_loss": history_train, "val_loss": history_val}, f, indent=2)
    print(f"Saved to {args.checkpoint_dir}. Predict with predict_vlm_price.py.")


if __name__ == "__main__":
    main()
