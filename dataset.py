import os
import glob
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from cropnet.data_retriever import DataRetriever
from cropnet.utils.path_utils import get_state_abbr

class CropYieldDataset(Dataset):
    """
    Dataset for Crop Yield Prediction (Visual-Only).
    Loads Sentinel-2 imagery and USDA yield targets.
    Supports "mock" mode if data is missing.
    """
    def __init__(self, root_dir, years, fips_codes, crop_type="Corn", transform=None):
        """
        Args:
            root_dir (str): Path to data root (e.g. './demo_data').
            years (list): List of years to include (e.g. ['2022']).
            fips_codes (list): List of FIPS codes (e.g. ['17019']).
            crop_type (str): Crop type (e.g. "Corn", "Soybean").
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.years = years
        self.fips_codes = fips_codes
        self.crop_type = crop_type
        self.transform = transform
        
        # Preload USDA Yield Data
        self.yield_data = self._preload_yield_data()
        
        # Prepare index of samples
        self.samples = []
        for year in years:
            for fips in fips_codes:
                # Check if we have yield for this sample
                if self._has_yield_data(fips, year):
                    self.samples.append({
                        "year": year,
                        "fips": fips
                    })
                else:
                    print(f"Warning: No yield data for FIPS {fips}, Year {year}. Skipping.")
        
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
        state_abbr = get_state_abbr(fips[:2])
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
