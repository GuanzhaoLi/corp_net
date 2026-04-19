#!/usr/bin/env python3
"""
Summarize temporal sampling inside an H5 (standalone or CropNet-style).

Standalone: keys "images" (T,C,H,W) and "dates" (T,).
CropNet-style: top-level FIPS groups; under each FIPS, date string groups with "data".

Usage:
  python inspect_h5_temporal.py /path/to/17113_2022.h5
  python inspect_h5_temporal.py /path/to/Sentinel/data/AG/2022/IL --pick-first-h5
  python inspect_h5_temporal.py cropnet_file.h5 --fips 17113
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import Counter
from datetime import datetime

import h5py
import numpy as np


def _decode_h5_date_elem(ds, i: int) -> str:
    """Match dataset_standalone: vlen str may come back as bytes."""
    if hasattr(ds, "asstr"):
        return str(ds.asstr()[i])
    v = ds[i]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _parse_date(s: str) -> datetime | None:
    s = str(s).strip()
    if len(s) >= 10:
        head = s[:10]
        try:
            return datetime.strptime(head, "%Y-%m-%d")
        except ValueError:
            pass
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d")
        except ValueError:
            pass
    return None


def _delta_days_sorted(dates: list[datetime]) -> list[int]:
    out = []
    for i in range(1, len(dates)):
        d = (dates[i] - dates[i - 1]).days
        out.append(d)
    return out


def inspect_standalone(path: str) -> None:
    with h5py.File(path, "r") as f:
        if "images" not in f or "dates" not in f:
            raise SystemExit(
                f"Standalone layout expected datasets 'images' and 'dates'. Keys: {list(f.keys())}"
            )
        images = f["images"]
        ds = f["dates"]
        T = int(images.shape[0])
        img_shape = tuple(images.shape)
        raw = [_decode_h5_date_elem(ds, i) for i in range(ds.shape[0])]
    parsed = [_parse_date(x) for x in raw]
    bad = sum(1 for p in parsed if p is None)
    dates_ok = [p for p in parsed if p is not None]

    print(f"File: {path}")
    print(f"Layout: standalone (images + dates)")
    print(f"T (frames): {T}, images shape: {img_shape}")
    print(f"First 5 dates (raw): {raw[:5]}")
    print(f"Last 5 dates (raw): {raw[-5:]}")
    print(f"Unparseable date strings: {bad} / {len(raw)}")

    if len(dates_ok) < 2:
        print("Not enough valid dates to compute gaps.")
        return

    deltas = _delta_days_sorted(dates_ok)
    ctr = Counter(deltas)
    most = ctr.most_common(8)
    print(f"Gap between consecutive frames (days): min={min(deltas)}, max={max(deltas)}, median={float(np.median(deltas)):.1f}")
    print(f"Most common gaps (days, count): {most}")

    span = (dates_ok[-1] - dates_ok[0]).days
    print(f"Calendar span (first→last valid date): {span} days, over {len(dates_ok)} frames (~{len(dates_ok) / max(span, 1) * 365:.1f} frames per 365d if linear)")


def inspect_cropnet(path: str, fips: str) -> None:
    fips = str(fips).zfill(5)
    with h5py.File(path, "r") as f:
        if fips not in f:
            keys = [k for k in f.keys() if str(k).isdigit() and len(str(k)) == 5]
            raise SystemExit(f"FIPS {fips} not in file. Example FIPS keys: {keys[:10]}")
        grp = f[fips]
        date_keys = [k for k in grp.keys() if k not in ("lat", "lon") and "data" in grp[k]]
        date_keys.sort()

    raw = list(date_keys)
    parsed = [_parse_date(x) for x in raw]
    bad = sum(1 for p in parsed if p is None)
    dates_ok = [p for p in parsed if p is not None]

    print(f"File: {path}")
    print(f"Layout: CropNet-style (FIPS / date / data)")
    print(f"FIPS: {fips}, T (frames): {len(raw)}")
    print(f"First 5 date keys: {raw[:5]}")
    print(f"Last 5 date keys: {raw[-5:]}")
    print(f"Unparseable date keys: {bad} / {len(raw)}")

    if len(dates_ok) < 2:
        print("Not enough valid dates to compute gaps.")
        return

    deltas = _delta_days_sorted(dates_ok)
    ctr = Counter(deltas)
    most = ctr.most_common(8)
    print(f"Gap between consecutive frames (days): min={min(deltas)}, max={max(deltas)}, median={float(np.median(deltas)):.1f}")
    print(f"Most common gaps (days, count): {most}")
    span = (dates_ok[-1] - dates_ok[0]).days
    print(f"Calendar span (first→last): {span} days, {len(dates_ok)} frames")


def detect_and_inspect(path: str, fips: str | None) -> None:
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
    if "images" in keys and "dates" in keys:
        inspect_standalone(path)
        return
    fips_keys = [k for k in keys if re.fullmatch(r"\d{5}", str(k))]
    if fips_keys:
        use = fips or fips_keys[0]
        inspect_cropnet(path, use)
        if fips is None and len(fips_keys) > 1:
            print(f"(Used first FIPS {use}; pass --fips to choose another.)")
        return
    raise SystemExit(f"Unknown H5 layout. Top-level keys: {keys[:20]}")


def main():
    p = argparse.ArgumentParser(description="Inspect temporal sampling in satellite H5")
    p.add_argument("path", help="Path to .h5 file, or directory when using --pick-first-h5")
    p.add_argument("--fips", default=None, help="FIPS for CropNet-style files (default: first numeric group)")
    p.add_argument(
        "--pick-first-h5",
        action="store_true",
        help="If path is a directory, open the first *.h5 found (non-recursive).",
    )
    args = p.parse_args()
    path = args.path
    if args.pick_first_h5:
        if not os.path.isdir(path):
            raise SystemExit("--pick-first-h5 requires a directory path")
        found = sorted(glob.glob(os.path.join(path, "*.h5")))
        if not found:
            found = sorted(glob.glob(os.path.join(path, "**", "*.h5"), recursive=True))
        if not found:
            raise SystemExit(f"No .h5 under {path}")
        path = found[0]
        print(f"Picked: {path}\n")

    if not os.path.isfile(path):
        raise SystemExit(f"Not a file: {path}")

    detect_and_inspect(path, args.fips)


if __name__ == "__main__":
    main()
