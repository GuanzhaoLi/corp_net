"""
Predict yield using trained VLM head + Qwen2-VL vision (Route B).
Usage:
  python predict_vlm_yield.py --data-dir ./standalone_data --checkpoint ./checkpoints_vlm/vlm_head_best.pth --out predictions_vlm.csv [--normalize-target]
"""
import argparse
import csv
import os
import json

import torch

from dataset_vlm_yield import VLMYieldDataset, DEFAULT_FRAME_INDICES
from model_vlm_yield import build_vlm_yield_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--checkpoint", required=True, help="vlm_head_best.pth or vlm_head_last.pth")
    parser.add_argument("--checkpoint-dir", default=None, help="For yield_norm.json if --normalize-target")
    parser.add_argument("--normalize-target", action="store_true")
    parser.add_argument("--out", default="predictions_vlm.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)
    yield_mean, yield_std = None, None
    if args.normalize_target:
        p = os.path.join(ckpt_dir, "yield_norm.json")
        if os.path.isfile(p):
            with open(p) as f:
                n = json.load(f)
            yield_mean, yield_std = n["mean"], n["std"]

    print("Loading model and head...")
    model = build_vlm_yield_model(num_frames=len(DEFAULT_FRAME_INDICES), device=device)
    state = torch.load(args.checkpoint, map_location=device)
    model.head.load_state_dict(state["head"], strict=True)
    model.eval()

    dataset = VLMYieldDataset(root_dir=args.data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            pred = model(images)
            pred = pred.cpu().float().numpy()
            if yield_mean is not None and yield_std is not None:
                pred = pred * yield_std + yield_mean
            for i in range(len(pred)):
                rows.append({
                    "fips": batch["fips"][i],
                    "year": batch["year"][i],
                    "predicted_yield_bu_per_acre": float(pred[i]),
                })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "predicted_yield_bu_per_acre"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
