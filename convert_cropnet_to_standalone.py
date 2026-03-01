"""
Convert existing CropNet-style H5 layout to standalone layout (one file per (fips, year)).
No cropnet import: we only read H5 under root_dir/Sentinel/data/AG/{year}/{state_abbr}/*.h5
and write standalone_data/images/{fips}_{year}.h5 with "images" (T,C,H,W) and "dates".

Usage:
  python convert_cropnet_to_standalone.py --root-dir ./demo_data --out-dir ./standalone_data
  Then copy your yields CSV to standalone_data/yields.csv (or pass --yields-csv).
"""
import os
import glob
import argparse
import h5py
import numpy as np

# State FIPS (first 2 digits) -> state abbr (same as dataset.py)
STATE_FIPS_TO_ABBR = {
    "17": "IL", "18": "IN", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
    "31": "NE", "38": "ND", "39": "OH", "46": "SD", "55": "WI",
    "19": "IA", "05": "AR", "01": "AL", "48": "TX", "51": "VA",
}


def state_abbr_from_fips(fips):
    return STATE_FIPS_TO_ABBR.get(str(fips)[:2], None)


def collect_fips_year_from_cropnet(root_dir, sentinel_subpath=None, fips_filter=None):
    """Scan cropnet layout and return set of (fips, year) that have data.
    Layout: root_dir/Sentinel/data/AG/{year}/{state_abbr}/*.h5
    If fips_filter is set (set or list of 5-char FIPS), only include those counties.
    """
    if sentinel_subpath is None:
        sentinel_subpath = os.path.join("Sentinel", "data", "AG")
    if fips_filter is not None:
        fips_filter = {str(f).zfill(5) for f in fips_filter}
    ag_path = os.path.join(root_dir, sentinel_subpath)
    if not os.path.isdir(ag_path):
        return set()
    found = set()
    for year_dir in os.listdir(ag_path):
        year_full = os.path.join(ag_path, year_dir)
        if not os.path.isdir(year_full) or not year_dir.isdigit():
            continue
        for state_dir in os.listdir(year_full):
            state_full = os.path.join(year_full, state_dir)
            if not os.path.isdir(state_full):
                continue
            for h5_path in glob.glob(os.path.join(state_full, "*.h5")):
                try:
                    with h5py.File(h5_path, "r") as f:
                        for key in f.keys():
                            if key in ("lat", "lon") or not str(key).isdigit() or len(str(key)) != 5:
                                continue
                            k = str(key)
                            if fips_filter is not None and k not in fips_filter:
                                continue
                            found.add((k, year_dir))
                except Exception:
                    continue
    return found


def extract_one_fips_year(cropnet_root, fips, year, sentinel_subpath=None):
    """
    Read all H5 under cropnet_root for this (fips, year), aggregate by date.
    Returns (images np array (T,C,H,W), list of date strings).
    """
    if sentinel_subpath is None:
        sentinel_subpath = os.path.join("Sentinel", "data", "AG")
    state_abbr = state_abbr_from_fips(fips)
    if not state_abbr:
        return None, None
    data_path = os.path.join(cropnet_root, sentinel_subpath, str(year), state_abbr)
    if not os.path.isdir(data_path):
        return None, None
    h5_files = glob.glob(os.path.join(data_path, "*.h5"))
    time_series = []
    for h5_f in h5_files:
        try:
            with h5py.File(h5_f, "r") as f:
                if fips not in f:
                    continue
                grp = f[fips]
                for date_str in grp.keys():
                    if date_str in ("lat", "lon"):
                        continue
                    if "data" not in grp[date_str]:
                        continue
                    data_np = grp[date_str]["data"][:]
                    n_tiles = data_np.shape[0]
                    img = data_np[n_tiles // 2]
                    img = img.astype(np.float32) / 255.0
                    if img.shape[-1] == 3:
                        img = np.transpose(img, (2, 0, 1))
                    time_series.append((date_str, img))
        except Exception:
            continue
    if not time_series:
        return None, None
    time_series.sort(key=lambda x: x[0])
    dates = [x[0] for x in time_series]
    images = np.stack([x[1] for x in time_series], axis=0).astype(np.float32)
    return images, dates


def main():
    parser = argparse.ArgumentParser(
        description="Convert CropNet H5 layout to standalone (one file per fips,year). "
        "Reuse existing per-state downloads: pass --fips or --fips-from-config to only convert your 11 counties."
    )
    parser.add_argument("--root-dir", default="./demo_data", help="CropNet data root (contains Sentinel/data/AG/...)")
    parser.add_argument("--out-dir", default="./standalone_data", help="Output root; images go to out_dir/images/")
    parser.add_argument("--sentinel-subpath", default=None, help='Default: Sentinel/data/AG')
    parser.add_argument("--yields-csv", default=None, help="If set, copy this CSV to out_dir/yields.csv")
    parser.add_argument(
        "--fips",
        nargs="*",
        default=None,
        help="Only convert these FIPS (e.g. 17113 18011 26063 ...). Default: convert all FIPS found in H5.",
    )
    parser.add_argument(
        "--fips-from-config",
        action="store_true",
        help="Use Config.FIPS_CODES as the FIPS filter (same 11 counties as config.py).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    fips_filter = None
    if args.fips_from_config:
        try:
            from config import Config
            fips_filter = Config.FIPS_CODES
            print(f"Using FIPS from config: {fips_filter}")
        except Exception as e:
            raise SystemExit(f"--fips-from-config failed: {e}")
    elif args.fips:
        fips_filter = [str(f).zfill(5) for f in args.fips]
        print(f"Using FIPS from CLI: {fips_filter}")

    pairs = collect_fips_year_from_cropnet(root_dir, args.sentinel_subpath, fips_filter=fips_filter)
    print(f"Found {len(pairs)} (fips, year) in cropnet layout under {root_dir}")

    written = 0
    for (fips, year) in sorted(pairs):
        images, dates = extract_one_fips_year(root_dir, fips, year, args.sentinel_subpath)
        if images is None:
            continue
        out_path = os.path.join(images_dir, f"{fips}_{year}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("images", data=images)
            dt = h5py.special_dtype(vlen=str)
            dset = f.create_dataset("dates", (len(dates),), dtype=dt)
            dset[:] = dates
        written += 1
    print(f"Wrote {written} standalone H5 files under {images_dir}")

    if args.yields_csv and os.path.isfile(args.yields_csv):
        import shutil
        dest = os.path.join(out_dir, "yields.csv")
        shutil.copy2(args.yields_csv, dest)
        print(f"Copied yields to {dest}")
    else:
        print("Add yields.csv under out_dir (fips, year, actual_yield_bu_per_acre) for training.")


if __name__ == "__main__":
    main()
