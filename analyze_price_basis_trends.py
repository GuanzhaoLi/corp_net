"""
Per-year crop price + price-basis trend table and extrinsic metrics.

Ground truth: macro CSV (same as training) — annual crop price P_t (last month of year),
price basis target ln(P_t / P_{t-1}). County predictions are pooled to US-level by mean per year.

Extrinsic metrics (series across years, aligned to years present in predictions):
  - Pearson r between actual basis and US-mean predicted basis (each model)
  - Sign agreement: fraction of years where sign(actual basis) == sign(US-mean pred)
  - Optional: chain predicted basis forward from true P in anchor year to compare level path

Usage:
  python analyze_price_basis_trends.py --data-dir ./standalone_data \\
    --vlm-pred predictions_vlm_price.csv \\
    --standalone-pred predictions_standalone.csv \\
    --price-checkpoint-dir ./checkpoints_vlm_price

  python analyze_price_basis_trends.py ... --out-csv price_trend_table.csv --out-latex price_trend_rows.tex
"""
import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from dataset_standalone import StandaloneCropYieldDataset


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _sign_agreement(actual: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of years where sign(actual)==sign(pred), ignoring pairs with either side ~0."""
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(actual) & np.isfinite(pred)
    sa = np.sign(actual[m])
    sp = np.sign(pred[m])
    ok = (sa != 0) | (sp != 0)
    if not ok.any():
        return float("nan")
    return float((sa[ok] == sp[ok]).mean())


def _chain_levels(years_sorted: list[str], pred_basis: dict[str, float], true_P: dict[str, float]) -> dict[str, float]:
    """
    P_hat[y0] = true P[y0]; P_hat[y] = P_hat[y-1] * exp(pred_basis[y]) for y > y0 in sorted order.
    Only for years where pred_basis and previous P_hat exist.
    """
    out = {}
    if not years_sorted:
        return out
    y0 = years_sorted[0]
    if y0 not in true_P or y0 not in pred_basis:
        return out
    out[y0] = true_P[y0]
    for y in years_sorted[1:]:
        py = str(int(y))
        pm1 = str(int(y) - 1)
        if pm1 not in out or py not in pred_basis:
            continue
        out[py] = out[pm1] * float(np.exp(pred_basis[py]))
    return out


def load_us_mean_by_year(csv_path: Optional[str]) -> dict[str, float]:
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(str)
    g = df.groupby("year")["predicted_price_basis"].mean()
    return {str(k): float(v) for k, v in g.items()}


def main():
    parser = argparse.ArgumentParser(description="Per-year price trend vs pooled predictions + extrinsic metrics.")
    parser.add_argument("--data-dir", default="./standalone_data")
    parser.add_argument("--vlm-pred", default=None, help="predictions_vlm_price.csv (optional)")
    parser.add_argument("--standalone-pred", default=None, help="predictions_standalone.csv (optional)")
    parser.add_argument("--macro-data-csv", default="macro_data.csv")
    parser.add_argument("--crop", choices=["soybean", "corn"], default=None)
    parser.add_argument("--price-checkpoint-dir", default=None, help="vlm_price_meta.json for crop if --crop omitted")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-latex", default=None, help="Append tabular rows (no preamble)")
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
    if crop is None:
        crop = "soybean"

    kwargs = dict(root_dir=args.data_dir, crop_type=crop, macro_data_csv_name=args.macro_data_csv)
    if macro_features is not None:
        kwargs["macro_features"] = macro_features
    ds = StandaloneCropYieldDataset(**kwargs)

    vlm_us = load_us_mean_by_year(args.vlm_pred)
    st_us = load_us_mean_by_year(args.standalone_pred)

    years_all = sorted(
        {y for y in ds.price_basis_lookup if not pd.isna(ds.price_basis_lookup[y])},
        key=lambda x: int(x),
    )
    if vlm_us or st_us:
        pred_years = set(vlm_us.keys()) | set(st_us.keys())
        years_table = sorted([y for y in years_all if y in pred_years], key=lambda x: int(x))
    else:
        years_table = years_all

    rows = []
    for y in years_table:
        act = float(ds.price_basis_lookup[y])
        pt = ds.crop_price_lookup.get(y, np.nan)
        ptm1 = ds.reference_crop_price_lookup.get(y, np.nan)
        rows.append(
            {
                "year": int(y),
                "P_t": pt,
                "P_t_minus_1": ptm1,
                "actual_ln_P_ratio": act,
                "pred_us_mean_vlm": vlm_us.get(y, np.nan),
                "pred_us_mean_standalone": st_us.get(y, np.nan),
                "err_vlm": (vlm_us.get(y, np.nan) - act) if y in vlm_us else np.nan,
                "err_standalone": (st_us.get(y, np.nan) - act) if y in st_us else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No overlapping years for table.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Per-year crop price and price basis (macro ground truth) ===\n")
    pd.set_option("display.max_rows", 30)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string(index=False))

    y = df["actual_ln_P_ratio"].to_numpy()
    yv = df["pred_us_mean_vlm"].to_numpy()
    ys = df["pred_us_mean_standalone"].to_numpy()

    print("\n=== Extrinsic metrics (US-mean prediction vs actual basis, across years in table) ===\n")
    print(f"  Years in table: {len(df)}  (crop={crop})")

    if np.isfinite(yv).sum() >= 2:
        print(f"  VLM:        r(actual, pred_us) = {_safe_corr(y, yv):.4f}   MAE = {np.nanmean(np.abs(yv - y)):.4f}")
        print(f"              sign(actual) vs sign(pred): {_sign_agreement(y, yv):.2%}")
    else:
        print("  VLM:        (no prediction CSV or insufficient years)")

    if np.isfinite(ys).sum() >= 2:
        print(f"  Standalone: r(actual, pred_us) = {_safe_corr(y, ys):.4f}   MAE = {np.nanmean(np.abs(ys - y)):.4f}")
        print(f"              sign(actual) vs sign(pred): {_sign_agreement(y, ys):.2%}")
    else:
        print("  Standalone: (no prediction CSV or insufficient years)")

    # Chained price level from true anchor at first table year
    years_str = [str(int(r)) for r in df["year"].tolist()]
    true_P = {str(k): float(ds.crop_price_lookup[k]) for k in ds.crop_price_lookup}
    vlm_b = {str(int(r["year"])): r["pred_us_mean_vlm"] for _, r in df.iterrows() if np.isfinite(r["pred_us_mean_vlm"])}
    st_b = {str(int(r["year"])): r["pred_us_mean_standalone"] for _, r in df.iterrows() if np.isfinite(r["pred_us_mean_standalone"])}

    if len(years_str) >= 2:
        print("\n=== Chained price level (extrinsic): anchor P_t at first year = truth; forward with exp(pred basis) ===\n")
        y0 = years_str[0]
        chain_v = _chain_levels(years_str, vlm_b, true_P)
        chain_s = _chain_levels(years_str, st_b, true_P)
        y_end = years_str[-1]
        if y_end in true_P:
            true_end = true_P[y_end]
            if y_end in chain_v:
                print(f"  Last year {y_end}: true P_t = {true_end:.2f}   VLM-chained P_t = {chain_v[y_end]:.2f}   rel. err = {(chain_v[y_end]/true_end - 1)*100:.2f}%")
            if y_end in chain_s:
                print(f"  Last year {y_end}: true P_t = {true_end:.2f}   Standalone-chained P_t = {chain_s[y_end]:.2f}   rel. err = {(chain_s[y_end]/true_end - 1)*100:.2f}%")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")

    if args.out_latex:
        lines = []
        for _, r in df.iterrows():
            def fnum(x):
                return "" if (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

            lines.append(
                f"{int(r['year'])} & {fnum(r['P_t'])} & {fnum(r['P_t_minus_1'])} & {r['actual_ln_P_ratio']:.4f} & "
                f"{fnum(r['pred_us_mean_vlm'])} & {fnum(r['pred_us_mean_standalone'])} \\\\"
            )
        with open(args.out_latex, "w") as f:
            f.write("% columns: year, P_t, P_{t-1}, actual ln ratio, pred US VLM, pred US standalone\n")
            f.write("\n".join(lines))
        print(f"Wrote LaTeX rows to {args.out_latex}")


if __name__ == "__main__":
    main()
