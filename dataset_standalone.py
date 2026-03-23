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
from datetime import datetime
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
            images = f["images"][:] # pull time series of satellite images; shape: (T, C, H, W) = (Time, Channel, Height, Width), where time is the number of dates with data for this (fips, year)
            if "dates" in f:
                ds = f["dates"] # pull date strings for each satellite image in the time series; shape: (T,)
                if hasattr(ds, "asstr"):
                    dates = [str(ds[i]) for i in range(ds.shape[0])]
                else:
                    dates = [str(ds[i]) for i in range(len(ds))]
            else:
                dates = [f"t{i}" for i in range(images.shape[0])]
        # Assume (T, H, W, C) or (T, C, H, W)
        if images.ndim == 4 and images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2)) # reorder to (T, C, H, W) = (Time, Channel, Height, Width). channel: 3 for RGB, 1 for greyscale, etc.
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0 # normalize pixel values that are [0,255] to [0,1]
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


def convert_monthly_macro_to_yearly(macro_df, crop_price_col, features):
    if "date" not in macro_df.columns:
        raise ValueError("Macro Data CSV must have 'date' column for monthly to yearly conversion")

    """Convert monthly macroeconomic data to yearly by taking the mean of each month for each year."""
    macro_df["year"] = macro_df["date"].map(lambda dt: str(datetime.strptime(dt, "%Y-%m-%d").year))

    # sort macro_df by date to ensure the "last" aggregation for crop price is the crop price of the final month of the year
    # this assumes that the monthly data is complete or mostly complete, so the last month of the year is a good approximation of the final crop price for that year
    macro_df = macro_df.sort_values(by='date', key=lambda x: pd.to_datetime(x, format="%Y-%m-%d"))

    # assuming average macro conditions over the year to predict year end crop prices with satellite image time series over the year
    # thus, take mean of macro features over the year and take the last crop price of the year as the target price for that year since it reflects the final market conditions after all growing season events have unfolded
    yearly_macro_df = macro_df.groupby("year")[features + [crop_price_col]].agg(
        {crop_price_col: "last"} | {feat: "mean" for feat in features}
    )
    return yearly_macro_df.reset_index(drop=False)


class StandaloneCropYieldDataset(Dataset):
    """
    Dataset that reads only from standalone_data/ layout. No cropnet.
    Samples are (fips, year) that have both yields in yields.csv and images in images/.
    """
    def __init__(
        self, root_dir, yields_csv_name="yields.csv", image_subdir="images", 
        macro_data_csv_name="macro_data.csv", macro_features=["crude_oil_usd", "usd_index", "fed_funds_rate", "cpi_yoy", "soybean_corn_ratio"], crop_type="soybean", 
        transform=None
    ):
        """
        Args:
            root_dir: e.g. ./standalone_data
            yields_csv_name: CSV under root_dir with fips, year, actual_yield_bu_per_acre (or yield_bu_per_acre)
            image_subdir: subdir under root_dir for images (e.g. "images")
            macro_data_csv_name: CSV under root_dir with monthly macroeconomic data
            macro_features: list of macro features to use
            crop_type: "soybean" or "corn" (crop price to predict)
            transform: optional callable (not used by default; use AugmentWrapperStandalone for train)
        """
        self.root_dir = os.path.abspath(root_dir)
        self.image_subdir = image_subdir
        self.macro_data_csv_name = macro_data_csv_name
        self.macro_features = macro_features
        self.crop_type = crop_type
        self.transform = transform

        # read in yield_bu_per_acre - yearly yield for each fips
        yield_csv_path = os.path.join(self.root_dir, yields_csv_name)
        if not os.path.isfile(yield_csv_path):
            raise FileNotFoundError(f"Yields CSV not found: {yield_csv_path}")

        df = pd.read_csv(yield_csv_path)
        for col in ("actual_yield_bu_per_acre", "yield_bu_per_acre", "predicted_yield_bu_per_acre"):
            if col in df.columns:
                yield_col = col
                break
        else:
            raise ValueError(f"CSV must have one of: actual_yield_bu_per_acre, yield_bu_per_acre. Got: {list(df.columns)}")

        # read in macroeconomic data - monthly crude_oil_usd, usd_index, fed_funds_rate, cpi_yoy, soy_corn_ratio
        macro_data_path = os.path.join(self.root_dir, self.macro_data_csv_name)
        if os.path.isfile(macro_data_path):
            self.raw_macro_data = pd.read_csv(macro_data_path)
        else:
            self.raw_macro_data = None
            raise FileNotFoundError(f"Macro Data CSV not found: {macro_data_path}")

        # (fips, year) -> yield_bu_per_acre (per yield_col) - for fips/year that have valid yield in yields_csv_name CSV
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

        # check that macro data has the necessary crop price column for the specified crop_type. The column should be "{crop_type}_price", e.g. "soybean_price" or "corn_price".
        price_col = f"{self.crop_type}_price"
        if price_col not in self.raw_macro_data.columns:
            raise ValueError(f"Macro data CSV must have column {price_col} for crop_type={self.crop_type}")

        # convert monthly macro data to yearly -> macro features (crude_oil_usd, usd_index, fed_funds_rate, cpi_yoy, soy_corn_ratio)
        self.yearly_macro_data = convert_monthly_macro_to_yearly(self.raw_macro_data, crop_price_col=price_col, features=self.macro_features)

        # crop_price_lookup: raw crop price for the year --- (year) -> crop price
        # crop_reference_price_lookup: uses the prior year raw crop price as the reference price --- (year) -> reference crop price
        # used to compute price basis (which will be the prediction target) = ln(price / reference_price) to mitigate long-term trends and focus on predicting the relative price changes due to yearly conditions
        # price_basis_lookup: (year) -> ln(price / reference_price)
        # where reference_price is the prior year's price, so price basis reflects the year-over-year change in crop price
        self.crop_price_lookup = {}
        self.reference_crop_price_lookup = {}
        self.price_basis_lookup = {}
        for _, row in self.yearly_macro_data.iterrows():
            y = row.get("year")
            if pd.isna(y):
                continue
            y = str(int(y))
            price = row.get(price_col)
            if pd.notna(price):
                self.crop_price_lookup[y] = float(price)
            
            prior_y = str(int(y) - 1)
            if prior_y in self.yearly_macro_data["year"].values:
                prior_price = self.yearly_macro_data.loc[self.yearly_macro_data["year"] == prior_y, price_col].values[0]
                if pd.notna(prior_price):
                    self.reference_crop_price_lookup[y] = float(prior_price)
                    if price > 0 and prior_price > 0:
                        self.price_basis_lookup[y] = float(np.log(float(price) / float(prior_price)))
                    else:
                        self.price_basis_lookup[y] = np.nan
                else:
                    self.reference_crop_price_lookup[y] = np.nan
                    self.price_basis_lookup[y] = np.nan
            else:
                self.reference_crop_price_lookup[y] = np.nan
                self.price_basis_lookup[y] = np.nan

        # list of {"fips": fips, "year": year} that have both yield in yields_csv_name CSV and images in image_subdir
        # this does not load the satellite images, just checks for their existence, so __len__ and __getitem__ only see valid samples with both yield and images.
        # only include samples that have valid price basis (i.e. both current and prior year crop price are available and > 0) since the price basis will be the prediction target for the model
        self.samples = []
        for (fips, year), _ in self.yield_lookup.items():
            if year in self.price_basis_lookup and not pd.isna(self.price_basis_lookup[year]):
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
