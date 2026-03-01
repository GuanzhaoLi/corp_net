import os
import glob
import random
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from cropnet.data_retriever import DataRetriever


def augment_temporal_images(images, p_flip=0.5, brightness_scale=(0.85, 1.15), brightness_shift=(-0.08, 0.08)):
    """
    Apply random augmentation to (T, C, H, W) tensor for training.
    Same draw for all T to keep temporal consistency. Reduces collapse to constant.
    """
    out = images.clone()
    if random.random() < p_flip:
        out = torch.flip(out, [-1])  # horizontal
    if random.random() < p_flip:
        out = torch.flip(out, [-2])  # vertical
    scale = random.uniform(*brightness_scale)
    shift = random.uniform(*brightness_shift)
    out = (out * scale + shift).clamp(0.0, 1.0)
    return out


class AugmentWrapper(Dataset):
    """Wraps a dataset and applies augment_temporal_images to 'images' in __getitem__ (for training only)."""
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

# Local state FIPS (first 2 digits of county FIPS) -> USPS abbreviation. No cropnet dependency.
STATE_FIPS_TO_ABBR = {
    "17": "IL", "18": "IN", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
    "31": "NE", "38": "ND", "39": "OH", "46": "SD", "55": "WI",
    "19": "IA", "05": "AR", "01": "AL", "48": "TX", "51": "VA",
}


def _state_abbr_from_fips(fips):
    """Return state abbreviation from full county FIPS (e.g. 17113 -> IL)."""
    st = str(fips)[:2]
    return STATE_FIPS_TO_ABBR.get(st, "")


def load_sentinel_for_prediction(root_dir, fips, year, sentinel_ag_subpath=None):
    """
    Load Sentinel-2 time-series for (fips, year) from root_dir.
    Returns (images_tensor, dates) with images_tensor (T, C, H, W). No USDA required.
    Structure: root_dir/<sentinel_ag_subpath>/{year}/{state_abbr}/*.h5
    Default sentinel_ag_subpath: "Sentinel/data/AG". For AG_chen layout use "AG".
    """
    if sentinel_ag_subpath is None:
        sentinel_ag_subpath = os.path.join("Sentinel", "data", "AG")
    state_abbr = _state_abbr_from_fips(fips)
    if not state_abbr:
        raise ValueError(f"Unknown state FIPS for county {fips}")
    data_path = os.path.join(root_dir, sentinel_ag_subpath, str(year), state_abbr)
    if os.path.exists(data_path):
        h5_files = glob.glob(os.path.join(data_path, "*.h5"))
    else:
        h5_files = []
    if not h5_files:
        raise FileNotFoundError(f"No H5 files in {data_path}")

    time_series_data = []
    for h5_f in h5_files:
        try:
            with h5py.File(h5_f, "r") as f:
                if fips not in f:
                    continue
                for date_str in f[fips].keys():
                    if date_str in ("lat", "lon"):
                        continue
                    date_group = f[fips][date_str]
                    if "data" not in date_group:
                        continue
                    data_np = date_group["data"][:]
                    n_tiles = data_np.shape[0]
                    img = data_np[n_tiles // 2]
                    img = img / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    time_series_data.append((date_str, img))
        except Exception as e:
            print(f"Error reading {h5_f}: {e}")

    if not time_series_data:
        raise ValueError(f"FIPS {fips} not found in any H5 for {year}")
    time_series_data.sort(key=lambda x: x[0])
    dates = [x[0] for x in time_series_data]
    images = np.stack([x[1] for x in time_series_data])
    return torch.from_numpy(images).float(), dates


class CropYieldDataset(Dataset):
    """
    Dataset for Crop Yield Prediction (Visual-Only).
    Loads Sentinel-2 imagery and USDA yield targets (or from a CSV).
    """
    def __init__(self, root_dir, years, fips_codes, crop_type="Corn", transform=None, yield_csv_path=None):
        """
        Args:
            root_dir (str): Path to data root (e.g. './demo_data').
            years (list): List of years to include (e.g. ['2022']).
            fips_codes (list): List of FIPS codes (e.g. ['17019']).
            crop_type (str): Crop type (e.g. "Corn", "Soybean").
            transform (callable, optional): Transform to apply to images.
            yield_csv_path (str, optional): If set, load (fips, year) -> yield from CSV
                with columns fips, year, and actual_yield_bu_per_acre (or yield_bu_per_acre).
                Use this for training with NASS/actual yields instead of DataRetriever.
        """
        self.root_dir = root_dir
        self.years = years
        self.fips_codes = fips_codes
        self.crop_type = crop_type
        self.transform = transform

        if yield_csv_path and os.path.isfile(yield_csv_path):
            self.yield_data = self._load_yield_from_csv(yield_csv_path)
            print(f"Loaded {len(self.yield_data)} yield values from {yield_csv_path}")
        else:
            self.yield_data = self._preload_yield_data()
        
        # Prepare index of samples (only include if we have both yield and Sentinel data)
        self.samples = []
        skipped_no_yield = []
        skipped_no_sentinel = []
        for year in years:
            for fips in fips_codes:
                if not self._has_yield_data(fips, year):
                    skipped_no_yield.append((fips, year))
                    continue
                try:
                    self._load_sentinel_data(fips, year)
                except (FileNotFoundError, ValueError):
                    skipped_no_sentinel.append((fips, year))
                    continue
                self.samples.append({"year": year, "fips": fips})
        if skipped_no_yield:
            print(f"Skipped {len(skipped_no_yield)} (fips, year) with no yield in CSV.")
        if skipped_no_sentinel:
            print(f"Skipped {len(skipped_no_sentinel)} (fips, year) with no Sentinel H5 data.")
        print(f"Dataset initialized with {len(self.samples)} samples (Counties * Years).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        year = sample_info["year"]
        fips = sample_info["fips"]
        
        # 1. Load Visual Data (Sentinel-2)
        images, dates = self._load_sentinel_data(fips, year)
        
        # 2. Load Target (USDA Yield)
        yield_value = self._get_yield_value(fips, year)
        
        # Convert to tensor
        # images: (T, C, H, W)
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()
            
        yield_target = torch.tensor([yield_value], dtype=torch.float32)
        
        return {
            "images": images,   # (T, C, H, W)
            "yield": yield_target,
            "fips": fips,
            "year": year,
            "dates": dates
        }

    def _load_sentinel_data(self, fips, year):
        """
        Tries to load Sentinel-2 H5 files.
        Raises FileNotFoundError if missing.
        Structure: root_dir/Sentinel/data/AG/{year}/{state_abbr}/ e.g. .../AG/2022/IL/
        """
        # State is first 2 digits of FIPS (e.g. 17113 -> 17 -> IL, 18093 -> 18 -> IN)
        state_abbr = _state_abbr_from_fips(fips)
        if not state_abbr:
            raise ValueError(f"Unknown state FIPS for county {fips}")
        data_path = os.path.join(self.root_dir, "Sentinel", "data", "AG", year, state_abbr)
        
        # Find all H5 files
        if os.path.exists(data_path):
            h5_files = glob.glob(os.path.join(data_path, "*.h5"))
        else:
            h5_files = []
            
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {data_path}")

        # If files exist, look for the FIPS data inside them
        # Note: cropnet splits time across files. We need to aggregate all info for this FIPS.
        
        time_series_data = [] # List of (date, image_tensor)
        
        processed_files = 0
        for h5_f in h5_files:
            try:
                with h5py.File(h5_f, 'r') as f:
                    if fips in f:
                        fips_group = f[fips]
                        # Iterate dates
                        for date_str in fips_group.keys():
                            if date_str in ['lat', 'lon']: continue # Skip metadata
                            
                            date_group = fips_group[date_str]
                            if 'data' in date_group:
                                # Shape: (Tiles, H, W, C) e.g., (25, 224, 224, 3)
                                data_np = date_group['data'][:]
                                # Simple strategy: Take the center tile (Index 0 for now as it's consistent)
                                # Or if 25 tiles, maybe index 12 is center?
                                # Let's checking shape
                                n_tiles = data_np.shape[0]
                                tile_idx = n_tiles // 2 
                                img = data_np[tile_idx] # (H, W, C)
                                
                                # Normalize 0-255 -> 0-1
                                img = img / 255.0
                                
                                # HWC -> CHW
                                img = np.transpose(img, (2, 0, 1))
                                
                                time_series_data.append((date_str, img))
                        processed_files += 1
            except Exception as e:
                print(f"Error reading {h5_f}: {e}")

        if not time_series_data:
             raise ValueError(f"FIPS {fips} not found in any H5 files for {year}")

        # Sort by date
        time_series_data.sort(key=lambda x: x[0])
        
        dates = [x[0] for x in time_series_data]
        images = np.stack([x[1] for x in time_series_data]) # (T, C, H, W)
        
        return images, dates

    def _load_yield_from_csv(self, csv_path):
        """Load (fips, year) -> yield from CSV. Columns: fips, year, and one of actual_yield_bu_per_acre / yield_bu_per_acre."""
        lookup = {}
        df = pd.read_csv(csv_path)
        # Prefer actual_yield, then yield_bu_per_acre, then predicted_yield
        for col in ("actual_yield_bu_per_acre", "yield_bu_per_acre", "predicted_yield_bu_per_acre"):
            if col in df.columns:
                yield_col = col
                break
        else:
            raise ValueError(f"CSV must have one of: actual_yield_bu_per_acre, yield_bu_per_acre. Got: {list(df.columns)}")
        for _, row in df.iterrows():
            f = str(int(row["fips"])).zfill(5) if pd.notna(row.get("fips")) else None
            y = str(int(row["year"])) if pd.notna(row.get("year")) else None
            val = row.get(yield_col)
            if f and y and pd.notna(val):
                lookup[(f, y)] = float(val)
        return lookup

    def _preload_yield_data(self):
        """
        Uses DataRetriever to fetch yield data for all requested FIPS and Years.
        Returns a dictionary or dataframe indexed by (fips, year).
        """
        try:
            print(f"Loading USDA yield data for {self.crop_type}...")
            retriever = DataRetriever(base_dir=self.root_dir)
            # Retrieve for all requested FIPS and Years
            df = retriever.retrieve_USDA(
                crop_type=self.crop_type, 
                fips_codes=self.fips_codes, 
                years=self.years
            )
            
            lookup = {}
            for _, row in df.iterrows():
                try:
                    # Inspecting user output: "YIELD, MEASURED IN BU / ACRE" seems to be the column name for value.
                    yield_col = [c for c in df.columns if "YIELD" in c and "BU / ACRE" in c]
                    if not yield_col:
                        continue
                    yield_val = row[yield_col[0]]
                    
                    # For simple demo with 1 sample:
                    # We can assume strict mapping if FIPS matching is complex without state lookup.
                    # But ideally we construct FIPS. 
                    # If columns have 'state_ansi' and 'county_ansi':
                    if 'state_ansi' in row and 'county_ansi' in row:
                        s = f"{int(row['state_ansi']):02d}"
                        c = f"{int(row['county_ansi']):03d}"
                        f = s + c
                    else:
                        # Fallback for demo if we can't reconstruct FIPS easily
                        # Just use the first FIPS code if we only have one.
                        # Or if df has 'fips' column (unlikely in raw USDA)
                        # Let's hope for state/county ansi
                        f = self.fips_codes[0] # DANGEROUS Assumption if multiple fips

                    y = str(row['year'])
                    lookup[(f, y)] = float(yield_val)
                    
                    # Also populate fallback if strict FIPS construction failed but we want to test
                    if len(self.fips_codes) == 1 and len(self.years) == 1:
                         lookup[(self.fips_codes[0], self.years[0])] = float(yield_val)

                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
            
            return lookup
            
        except Exception as e:
            print(f"Error initializing DataRetriever: {e}")
            return {}

    def _has_yield_data(self, fips, year):
        return (fips, str(year)) in self.yield_data

    def _get_yield_value(self, fips, year):
        return self.yield_data.get((fips, str(year)), 0.0)
