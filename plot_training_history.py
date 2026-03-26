"""
Plot train/val loss from training_history.json (written by train_standalone.py).

Usage:
  python plot_training_history.py --checkpoint-dir ./checkpoints_standalone
  python plot_training_history.py --history /path/to/training_history.json --out loss.png
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing training_history.json (default: ./checkpoints_standalone)",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Explicit path to training_history.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save figure to this path (png/pdf). If omitted, shows interactively.",
    )
    args = parser.parse_args()

    if args.history:
        path = args.history
    else:
        ckpt = args.checkpoint_dir or "./checkpoints_standalone"
        path = os.path.join(ckpt, "training_history.json")

    if not os.path.isfile(path):
        print(
            f"No file at {path}. Re-run training with an updated train_standalone.py, "
            "or pass --history pointing to a JSON with keys train_loss, val_loss (lists of floats).",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: conda install matplotlib  # or pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="Train loss", marker=".", markersize=4)
    plt.plot(epochs, val_loss, label="Val loss", marker=".", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
