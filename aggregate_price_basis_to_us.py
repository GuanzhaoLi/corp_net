"""
Aggregate county-level price-basis predictions (predict_standalone.py) to a US-level estimate.

Unlike yield (aggregate_to_us.py), price basis is a *year-level* market quantity in your setup:
the training target is the same for every county in the same year (macro + national price).
County predictions differ because of satellite. Pooling them is an ensemble / noise reduction.

Methods:
  mean          — unweighted average of predicted_price_basis over counties (default).
  median        — robust to outliers.
  acre_weighted — mean weighted by harvested acres per county (optional acres CSV).

Usage:
  python aggregate_price_basis_to_us.py predictions_standalone.csv --out us_price_basis.csv
  python aggregate_price_basis_to_us.py predictions_standalone.csv --method median --year 2023
  python aggregate_price_basis_to_us.py predictions_standalone.csv --method acre_weighted \\
    --acres-csv fips_harvested_acres_example.csv --out us_price_basis.csv
"""
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Pool county price-basis predictions into a US (national) estimate per year."
    )
    parser.add_argument(
        "predictions_csv",
        help="CSV from predict_standalone.py: fips, year, predicted_price_basis",
    )
    parser.add_argument(
        "--method",
        choices=("mean", "median", "acre_weighted"),
        default="mean",
        help="How to combine counties within a year (default: mean).",
    )
    parser.add_argument(
        "--acres-csv",
        default=None,
        help="Required for acre_weighted: fips, year, harvested_acres (same layout as aggregate_to_us.py).",
    )
    parser.add_argument("--year", default=None, help="Single year to report (default: all years in CSV).")
    parser.add_argument("--out", default=None, help="Write year, us_predicted_price_basis, n_counties, ...")
    args = parser.parse_args()

    rows = []
    with open(args.predictions_csv, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "predicted_price_basis" not in r.fieldnames:
            raise SystemExit(
                "CSV must have columns fips, year, predicted_price_basis "
                "(output of predict_standalone.py for CropPriceModel)."
            )
        for row in r:
            rows.append(
                {
                    "fips": row["fips"].strip(),
                    "year": str(row["year"]).strip(),
                    "v": float(row["predicted_price_basis"]),
                }
            )
    if not rows:
        raise SystemExit("No rows in predictions CSV.")

    acres = {}
    if args.method == "acre_weighted":
        if not args.acres_csv:
            raise SystemExit("--acres-csv is required when --method acre_weighted")
        with open(args.acres_csv, newline="") as f:
            ar = csv.DictReader(f)
            for row in ar:
                acres[(row["fips"].strip(), str(row["year"]).strip())] = float(row["harvested_acres"])

    by_year = defaultdict(list)
    for row in rows:
        by_year[row["year"]].append(row)

    years = sorted(by_year.keys())
    if args.year:
        years = [y for y in years if y == str(args.year)]
        if not years:
            raise SystemExit(f"No rows for year {args.year}")

    out_rows = []
    for year in years:
        items = by_year[year]
        vals = [x["v"] for x in items]
        n = len(vals)

        if args.method == "mean":
            us = sum(vals) / n
        elif args.method == "median":
            us = float(statistics.median(vals))
        else:
            wsum = 0.0
            wtot = 0.0
            missing = []
            for x in items:
                a = acres.get((x["fips"], year))
                if a is None or a <= 0:
                    missing.append(x["fips"])
                    continue
                wsum += x["v"] * a
                wtot += a
            if wtot <= 0:
                raise SystemExit(
                    f"Year {year}: no valid acres for acre_weighted. Missing FIPS (sample): {missing[:5]} ..."
                )
            us = wsum / wtot

        print(f"Year {year}: US pooled price basis ({args.method}) = {us:.6f}  (n_counties={n})")
        out_rows.append(
            {
                "year": year,
                "us_predicted_price_basis": round(us, 8),
                "n_counties": n,
                "method": args.method,
            }
        )

    if args.out and out_rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["year", "us_predicted_price_basis", "n_counties", "method"],
            )
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
