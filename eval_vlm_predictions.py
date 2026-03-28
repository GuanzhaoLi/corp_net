"""
Compare VLM prediction CSVs to labels in standalone_data.

Yield: merge predictions_vlm.csv with yields.csv on (fips, year); report MAE, RMSE, bias, R², correlation.

Price basis: target is year-level ln(P_t / P_{t-1}) from macro_data.csv (same as training).
  Rows are per (fips, year); truth repeats per county in a year. Reports:
  - row-level metrics (all merged rows)
  - per-year metrics (mean predicted vs one truth per year) — preferred summary

Usage:
  python eval_vlm_predictions.py --data-dir ./standalone_data \\
    --yield-pred predictions_vlm.csv \\
    --price-pred predictions_vlm_price.csv \\
    --price-checkpoint-dir ./checkpoints_vlm_price

  # Only yield or only price: omit the other --*-pred (or pass a missing path to skip with a warning)

  python eval_vlm_predictions.py --data-dir ./standalone_data --yield-pred predictions_vlm.csv

  # Standalone CNN price head (same CSV columns as VLM price):
  python eval_vlm_predictions.py --data-dir ./standalone_data \\
    --price-pred predictions_standalone.csv --skip-yield [--crop soybean]
"""
import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from dataset_standalone import StandaloneCropYieldDataset


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-20:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return float("nan")
    c = np.corrcoef(y_true, y_pred)[0, 1]
    return float(c) if not np.isnan(c) else float("nan")


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["year"].astype(str)
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    return out


def eval_yield(
    pred_path: str,
    data_dir: str,
    yields_csv_name: str,
) -> Optional[dict]:
    if not os.path.isfile(pred_path):
        print(f"[yield] Skip: file not found: {pred_path}", file=sys.stderr)
        return None
    pred = _norm_keys(pd.read_csv(pred_path))
    if "predicted_yield_bu_per_acre" not in pred.columns:
        print("[yield] Skip: CSV needs column predicted_yield_bu_per_acre", file=sys.stderr)
        return None

    ypath = os.path.join(data_dir, yields_csv_name)
    if not os.path.isfile(ypath):
        print(f"[yield] Skip: yields CSV not found: {ypath}", file=sys.stderr)
        return None
    act = _norm_keys(pd.read_csv(ypath))
    for col in ("actual_yield_bu_per_acre", "yield_bu_per_acre", "predicted_yield_bu_per_acre"):
        if col in act.columns:
            truth_col = col
            break
    else:
        print("[yield] Skip: yields CSV needs actual_yield_bu_per_acre or yield_bu_per_acre", file=sys.stderr)
        return None

    m = pred.merge(act[["fips", "year", truth_col]], on=["fips", "year"], how="inner")
    if len(m) == 0:
        print("[yield] No overlapping (fips, year) rows between prediction and yields CSV.", file=sys.stderr)
        return None

    y_hat = m["predicted_yield_bu_per_acre"].to_numpy()
    y = m[truth_col].to_numpy()
    err = y_hat - y
    out = {
        "n": int(len(m)),
        "mae_bu_acre": float(np.abs(err).mean()),
        "rmse_bu_acre": float(np.sqrt((err**2).mean())),
        "bias_bu_acre": float(err.mean()),
        "r2": _r2(y, y_hat),
        "corr": _corr(y, y_hat),
        "mean_actual_bu_acre": float(y.mean()),
    }
    print("\n=== Yield (bu/acre) ===")
    print(f"  n (merged rows):     {out['n']}")
    print(f"  MAE:                 {out['mae_bu_acre']:.4f}")
    print(f"  RMSE:                {out['rmse_bu_acre']:.4f}")
    print(f"  bias (pred - actual): {out['bias_bu_acre']:.4f}")
    print(f"  R²:                  {out['r2']:.4f}")
    print(f"  Pearson r:           {out['corr']:.4f}")
    print(f"  mean actual yield:   {out['mean_actual_bu_acre']:.4f}")
    return out


def eval_price_basis(
    pred_path: str,
    data_dir: str,
    crop: str,
    macro_data_csv_name: str,
    macro_features: Optional[List[str]],
) -> Optional[dict]:
    if not os.path.isfile(pred_path):
        print(f"[price_basis] Skip: file not found: {pred_path}", file=sys.stderr)
        return None
    pred = _norm_keys(pd.read_csv(pred_path))
    if "predicted_price_basis" not in pred.columns:
        print("[price_basis] Skip: CSV needs column predicted_price_basis", file=sys.stderr)
        return None

    kwargs = dict(
        root_dir=data_dir,
        crop_type=crop,
        macro_data_csv_name=macro_data_csv_name,
    )
    if macro_features is not None:
        kwargs["macro_features"] = macro_features
    try:
        ds = StandaloneCropYieldDataset(**kwargs)
    except Exception as e:
        print(f"[price_basis] Failed to load StandaloneCropYieldDataset: {e}", file=sys.stderr)
        return None

    lookup = ds.price_basis_lookup
    pred = pred.copy()
    pred["actual_price_basis"] = pred["year"].map(lambda y: lookup.get(y, np.nan))
    pred = pred.dropna(subset=["actual_price_basis"])
    if len(pred) == 0:
        print("[price_basis] No rows with a valid actual price_basis for prediction years.", file=sys.stderr)
        return None

    y_hat = pred["predicted_price_basis"].to_numpy(dtype=np.float64)
    y = pred["actual_price_basis"].to_numpy(dtype=np.float64)
    err = y_hat - y

    by_year = pred.groupby("year", sort=True).agg(
        pred_mean=("predicted_price_basis", "mean"),
        actual=("actual_price_basis", "first"),
    )
    by_year["err"] = by_year["pred_mean"] - by_year["actual"]
    ey = by_year["err"].to_numpy()

    out = {
        "n_rows": int(len(pred)),
        "n_years": int(pred["year"].nunique()),
        "crop": crop,
        "row_mae": float(np.abs(err).mean()),
        "row_rmse": float(np.sqrt((err**2).mean())),
        "row_bias": float(err.mean()),
        "row_r2": _r2(y, y_hat),
        "row_corr": _corr(y, y_hat),
        "per_year_mae": float(np.abs(ey).mean()),
        "per_year_rmse": float(np.sqrt((ey**2).mean())),
        "per_year_bias": float(ey.mean()),
        "per_year_r2": _r2(by_year["actual"].to_numpy(), by_year["pred_mean"].to_numpy()),
        "per_year_corr": _corr(by_year["actual"].to_numpy(), by_year["pred_mean"].to_numpy()),
    }
    print("\n=== Price basis (ln P_t / P_{t-1}, macro-derived) ===")
    print(f"  crop:                {crop}")
    print(f"  n prediction rows:   {out['n_rows']}  (unique years: {out['n_years']})")
    print("  Row-level (each county-year; truth repeats per year):")
    print(f"    MAE:   {out['row_mae']:.6f}   RMSE: {out['row_rmse']:.6f}   bias: {out['row_bias']:.6f}")
    print(f"    R²:    {out['row_r2']:.6f}   r: {out['row_corr']:.6f}")
    print("  Per-year (mean predicted vs one actual per year) — preferred summary:")
    print(f"    MAE:   {out['per_year_mae']:.6f}   RMSE: {out['per_year_rmse']:.6f}   bias: {out['per_year_bias']:.6f}")
    print(f"    R²:    {out['per_year_r2']:.6f}   r: {out['per_year_corr']:.6f}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM yield / price CSVs against standalone labels.")
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--yield-pred", default="predictions_vlm.csv")
    parser.add_argument("--price-pred", default="predictions_vlm_price.csv")
    parser.add_argument("--yields-csv", default="yields.csv")
    parser.add_argument("--macro-data-csv", default="macro_data.csv")
    parser.add_argument(
        "--price-checkpoint-dir",
        default=None,
        help="Directory containing vlm_price_meta.json (for crop and macro_features).",
    )
    parser.add_argument(
        "--crop",
        choices=["soybean", "corn"],
        default=None,
        help="Override crop for price-basis truth; default from vlm_price_meta.json or soybean.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Write all metrics to this JSON file.",
    )
    parser.add_argument("--skip-yield", action="store_true", help="Do not evaluate yield CSV.")
    parser.add_argument("--skip-price", action="store_true", help="Do not evaluate price_basis CSV.")
    args = parser.parse_args()

    crop = args.crop
    macro_features = None
    if args.price_checkpoint_dir:
        meta_path = os.path.join(args.price_checkpoint_dir, "vlm_price_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if crop is None:
                crop = meta.get("crop", "soybean")
            macro_features = meta.get("macro_features")
        elif crop is None:
            crop = "soybean"
    elif crop is None:
        crop = "soybean"

    results = {}
    results["yield"] = None if args.skip_yield else eval_yield(args.yield_pred, args.data_dir, args.yields_csv)
    results["price_basis"] = None
    if not args.skip_price:
        results["price_basis"] = eval_price_basis(
            args.price_pred,
            args.data_dir,
            crop,
            args.macro_data_csv,
            macro_features,
        )

    if args.out_json:
        # JSON-serializable only
        serial = {}
        for k, v in results.items():
            serial[k] = v if v is not None else None
        with open(args.out_json, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"\nWrote metrics to {args.out_json}")

    if results["yield"] is None and results["price_basis"] is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
