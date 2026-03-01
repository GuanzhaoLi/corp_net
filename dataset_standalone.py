"""
Standalone dataset: no cropnet dependency.
Reads from ./standalone_data/ layout:
  - standalone_data/images/{fips}_{year}.h5  with keys "images" (T,C,H,W), "dates" (optional)
  - standalone_data/yields.csv  with fips, year, actual_yield_bu_per_acre (or yield_bu_per_acre)
All paths are explicit and per (fips, year) so you can verify each sample.
"""
import os
import glob
import random
import json
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def augment_temporal_images(images, p_flip=0.5, brightness_scale=(0.85, 1.15), brightness_shift=(-0.08, 0.08)):
    """Same as dataset.py: random flip + brightness, applied to (T,C,H,W)."""
    out = images.clone()
    if random.random() < p_flip:
        out = torch.flip(out, [-1])
    if random.random() < p_flip:
        out = torch.flip(out, [-2])
    scale = random.uniform(*brightness_scale)
    shift = random.uniform(*brightness_shift)
    out = (out * scale + shift).clamp(0.0, 1.0)
    return out


class AugmentWrapperStandalone(Dataset):
    """Wraps StandaloneCropYieldDataset and applies augment_temporal_images to 'images'."""
    def __init__(self, dataset, augment_fn=None):
        self.dataset = dataset
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.augment_fn is not None and "images" in item:
            item = {**item, "images": self.augment_fn(item["images"])}
        return item


def load_sample_images(root_dir, fips, year, image_subdir="images"):
    """
    Load time-series images for (fips, year) from standalone layout.
    Returns (images_tensor, dates).
    Tries:
      1) {root_dir}/{image_subdir}/{fips}_{year}.h5  with "images", "dates"
      2) {root_dir}/{image_subdir}/{fips}/{year}.npy + {fips}/{year}_dates.json
    """
    base = os.path.join(root_dir, image_subdir)
    fips = str(fips).zfill(5)
    year = str(year)

    # Option 1: single H5 per (fips, year)
    h5_path = os.path.join(base, f"{fips}_{year}.h5")
    if os.path.isfile(h5_path):
        with h5py.File(h5_path, "r") as f:
            images = f["images"][:]
            if "dates" in f:
                ds = f["dates"]
                if hasattr(ds, "asstr"):
                    dates = [str(ds[i]) for i in range(ds.shape[0])]
                else:
                    dates = [str(ds[i]) for i in range(len(ds))]
            else:
                dates = [f"t{i}" for i in range(images.shape[0])]
        # Assume (T, H, W, C) or (T, C, H, W)
        if images.ndim == 4 and images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
        return torch.from_numpy(images).float(), dates

    # Option 2: npy + json
    npy_path = os.path.join(base, fips, f"{year}.npy")
    json_path = os.path.join(base, fips, f"{year}_dates.json")
    if os.path.isfile(npy_path):
        images = np.load(npy_path)
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
        if images.ndim == 4 and images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        if os.path.isfile(json_path):
            with open(json_path) as f:
                dates = json.load(f)
        else:
            dates = [f"t{i}" for i in range(images.shape[0])]
        return torch.from_numpy(images).float(), dates

    raise FileNotFoundError(f"No standalone image data for fips={fips} year={year}. Tried {h5_path} and {npy_path}")


class StandaloneCropYieldDataset(Dataset):
    """
    Dataset that reads only from standalone_data/ layout. No cropnet.
    Samples are (fips, year) that have both yields in yields.csv and images in images/.
    """
    def __init__(self, root_dir, yields_csv_name="yields.csv", image_subdir="images", transform=None):
        """
        Args:
            root_dir: e.g. ./standalone_data
            yields_csv_name: CSV under root_dir with fips, year, actual_yield_bu_per_acre (or yield_bu_per_acre)
            image_subdir: subdir under root_dir for images (e.g. "images")
            transform: optional callable (not used by default; use AugmentWrapperStandalone for train)
        """
        self.root_dir = os.path.abspath(root_dir)
        self.image_subdir = image_subdir
        self.transform = transform
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
            f = row.get("fips")
            y = row.get("year")
            if pd.isna(f) or pd.isna(y):
                continue
            f = str(int(f)).zfill(5)
            y = str(int(y))
            val = row.get(yield_col)
            if pd.notna(val):
                self.yield_lookup[(f, y)] = float(val)

        self.samples = []
        for (fips, year), _ in self.yield_lookup.items():
            try:
                load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
                self.samples.append({"fips": fips, "year": year})
            except FileNotFoundError:
                continue
        print(f"[StandaloneCropYieldDataset] {len(self.samples)} samples from {root_dir} (with both yields and images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        fips, year = info["fips"], info["year"]
        images, dates = load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
        y = self.yield_lookup[(fips, year)]
        return {
            "images": images,
            "yield": torch.tensor([y], dtype=torch.float32),
            "fips": fips,
            "year": year,
            "dates": dates,
        }
