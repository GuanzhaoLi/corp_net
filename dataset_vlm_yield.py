"""
Dataset for VLM yield (Route B): load standalone H5, subsample 5 key frames per (fips, year).
Returns (images [5, 3, 224, 224], yield). No cropnet dependency.

VLMPriceDataset: same 5 frames + yearly macro + price_basis target (for Qwen2VLPriceModel).
"""
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from dataset_standalone import (
    StandaloneCropYieldDataset,
    load_sample_images,
    build_national_monthly_macro_lookup,
    macro_sequence_for_dates,
)

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


class VLMPriceDataset(Dataset):
    """
    Same subsampled 5 frames as VLMYieldDataset, plus macro vector and price_basis target.
    Reuses StandaloneCropYieldDataset for macro/price_basis/yield CSV logic (needs macro_data.csv).
    """

    def __init__(
        self,
        root_dir,
        yields_csv_name="yields.csv",
        macro_data_csv_name="macro_data.csv",
        macro_features=None,
        image_subdir="images",
        crop_type="soybean",
        frame_indices=None,
        use_monthly_national_macro=False,
        national_monthly_csv_path=None,
    ):
        macro_features = macro_features or [
            "crude_oil_usd",
            "usd_index",
            "fed_funds_rate",
            "cpi_yoy",
            "soybean_corn_ratio",
        ]
        self.frame_indices = frame_indices or list(DEFAULT_FRAME_INDICES)
        self.use_monthly_national_macro = bool(use_monthly_national_macro)
        self.base = StandaloneCropYieldDataset(
            root_dir=root_dir,
            yields_csv_name=yields_csv_name,
            macro_data_csv_name=macro_data_csv_name,
            macro_features=macro_features,
            image_subdir=image_subdir,
            crop_type=crop_type,
        )
        self.root_dir = self.base.root_dir
        self.image_subdir = self.base.image_subdir
        self.macro_features = self.base.macro_features
        self.price_basis_lookup = self.base.price_basis_lookup
        self.yearly_macro_data = self.base.yearly_macro_data

        self.monthly_macro_lookup = None
        self._national_monthly_csv_resolved = None
        if self.use_monthly_national_macro:
            mp = national_monthly_csv_path or os.path.join(self.root_dir, macro_data_csv_name)
            mp = os.path.abspath(mp)
            if not os.path.isfile(mp):
                raise FileNotFoundError(f"National monthly macro CSV not found: {mp}")
            self.monthly_macro_lookup = build_national_monthly_macro_lookup(mp, list(self.macro_features))
            self._national_monthly_csv_resolved = mp

        self.samples = []
        for s in self.base.samples:
            fips, year = s["fips"], s["year"]
            try:
                images, _ = load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
            except FileNotFoundError:
                continue
            if max(self.frame_indices) >= images.shape[0]:
                continue
            self.samples.append({"fips": fips, "year": year})
        print(
            f"[VLMPriceDataset] {len(self.samples)} samples, {len(self.frame_indices)} frames "
            f"(macro_dim={len(self.macro_features)})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        fips, year = info["fips"], info["year"]
        images, dates = load_sample_images(self.root_dir, fips, year, image_subdir=self.image_subdir)
        sub = images[self.frame_indices]
        if not isinstance(sub, torch.Tensor):
            sub = torch.from_numpy(sub).float()
        pb = self.price_basis_lookup[year]
        if self.use_monthly_national_macro:
            sub_dates = [dates[i] for i in self.frame_indices]
            arr = macro_sequence_for_dates(
                year,
                sub_dates,
                self.macro_features,
                self.monthly_macro_lookup,
                len(self.macro_features),
            )
            macro = torch.from_numpy(arr).float()
        else:
            yr_match = self.yearly_macro_data["year"].astype(str) == str(year)
            macro_arr = self.yearly_macro_data.loc[yr_match, self.macro_features].values[0]
            macro = torch.tensor(macro_arr, dtype=torch.float32)
        return {
            "images": sub,
            "macro": macro,
            "price_basis": torch.tensor([pb], dtype=torch.float32),
            "fips": fips,
            "year": year,
        }
