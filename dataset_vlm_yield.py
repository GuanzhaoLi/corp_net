"""
Dataset for VLM yield (Route B): load standalone H5, subsample 5 key frames per (fips, year).
Returns (images [5, 3, 224, 224], yield). No cropnet dependency.
"""
import os
import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from dataset_standalone import load_sample_images

# Default: 5 frames evenly over 24 (indices 0, 6, 12, 18, 23)
DEFAULT_FRAME_INDICES = [0, 6, 12, 18, 23]


class VLMYieldDataset(Dataset):
    """
    (fips, year) -> 5 subsampled images (T=5, C, H, W) + yield.
    Images are float in [0, 1], shape (5, 3, 224, 224).
    """
    def __init__(self, root_dir, yields_csv_name="yields.csv", image_subdir="images", frame_indices=None):
        self.root_dir = os.path.abspath(root_dir)
        self.image_subdir = image_subdir
        self.frame_indices = frame_indices or DEFAULT_FRAME_INDICES
        csv_path = os.path.join(self.root_dir, yields_csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Yields CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        for col in ("actual_yield_bu_per_acre", "yield_bu_per_acre", "predicted_yield_bu_per_acre"):
            if col in df.columns:
                yield_col = col
                break
        else:
            raise ValueError(f"CSV must have one of: actual_yield_bu_per_acre, yield_bu_per_acre. Got: {list(df.columns)}")
        self.yield_lookup = {}
        for _, row in df.iterrows():
            f, y = row.get("fips"), row.get("year")
            if pd.isna(f) or pd.isna(y):
                continue
            f, y = str(int(f)).zfill(5), str(int(y))
            v = row.get(yield_col)
            if pd.notna(v):
                self.yield_lookup[(f, y)] = float(v)
        self.samples = []
        for (fips, year) in self.yield_lookup:
            try:
                images, _ = load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
            except FileNotFoundError:
                continue
            T = images.shape[0]
            if max(self.frame_indices) >= T:
                continue
            self.samples.append({"fips": fips, "year": year})
        print(f"[VLMYieldDataset] {len(self.samples)} samples, {len(self.frame_indices)} frames per sample")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        fips, year = info["fips"], info["year"]
        images, _ = load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
        # Subsample: (T,C,H,W) -> (5,C,H,W)
        sub = images[self.frame_indices]
        if not isinstance(sub, torch.Tensor):
            sub = torch.from_numpy(sub).float()
        y = self.yield_lookup[(fips, year)]
        return {
            "images": sub,
            "yield": torch.tensor([y], dtype=torch.float32),
            "fips": fips,
            "year": year,
        }
