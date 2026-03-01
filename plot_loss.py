"""Plot train/val loss from loss_log.txt (created during train.py)."""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to loss_log.txt (default: checkpoints/loss_log.txt)")
    parser.add_argument("--out", type=str, default="loss_curve.png", help="Output plot path")
    args = parser.parse_args()
    log_path = args.log or os.path.join("checkpoints", "loss_log.txt")
    if not os.path.isfile(log_path):
        print(f"Not found: {log_path}. Run train.py first to generate loss_log.txt")
        return
    with open(log_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        print("Need at least a header and one epoch row.")
        return
    header = lines[0].split("\t")
    epochs, train_losses, val_losses = [], [], []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            epochs.append(int(parts[0]))
            train_losses.append(float(parts[1]))
            val_losses.append(float(parts[2]))
    if not epochs:
        print("No epoch data in log.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    plt.plot(epochs, val_losses, "r-s", label="Val Loss", markersize=4)
    plt.title("Training and Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out)
    print(f"Saved {args.out} ({len(epochs)} epochs)")
    print(f"Train loss: {train_losses[0]:.2f} -> {train_losses[-1]:.2f}")
    print(f"Val loss:   {val_losses[0]:.2f} -> {val_losses[-1]:.2f}")

if __name__ == "__main__":
    main()
