"""
From county-level predicted yields (CSV from predict_batch.py) + harvested acres per county,
estimate US total soybean yield and production (area-weighted upscaling).

Why this makes sense:
  - Your 15 FIPS are a sample. If they are chosen to be representative of major producing
    regions, the area-weighted average yield of the sample is a reasonable estimator for
    national average yield (bu/acre). Then: US production ≈ US_yield_estimate × US_total_acres.
  - You need harvested acres per county (and US total) from USDA NASS Quick Stats or similar.
    Provide them in a CSV (see fips_harvested_acres_example.csv).
    Data source: USDA NASS Quick Stats (quickstats.nass.usda.gov) — commodity Soybeans, county level, "Area Harvested".
"""
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(
        description="Upscale county predictions to US soybean yield/production (area-weighted)."
    )
    parser.add_argument(
        "predictions_csv",
        type=str,
        help="CSV from predict_batch.py: fips, year, predicted_yield_bu_per_acre",
    )
    parser.add_argument(
        "acres_csv",
        type=str,
        help="CSV: fips, year, harvested_acres (soybean acres per county for that year)",
    )
    parser.add_argument(
        "--us-acres",
        type=float,
        default=None,
        help="US total soybean harvested acres (e.g. 86e6). If omitted, uses config.US_SOYBEAN_ACRES_2024 for year 2024.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Single year to report (default: report all years in predictions)",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Report US estimate for every year in predictions (default if CSV has multiple years)",
    )
    args = parser.parse_args()

    # Load predictions
    preds = {}
    with open(args.predictions_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            preds[(row["fips"], row["year"])] = float(row["predicted_yield_bu_per_acre"])
    if not preds:
        raise SystemExit("No rows in predictions CSV.")

    # Load acres (fips, year) -> harvested_acres
    acres = {}
    with open(args.acres_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row["fips"], row["year"])
            acres[k] = float(row["harvested_acres"])

    years_in_preds = sorted(set(y for (_, y) in preds))
    if args.year:
        years_to_report = [args.year]
    elif args.all_years or len(years_in_preds) > 1:
        years_to_report = years_in_preds
    else:
        years_to_report = [years_in_preds[0]]

    try:
        from config import Config
        us_acres_by_year = getattr(Config, "US_SOYBEAN_ACRES_BY_YEAR", None)
        us_acres_2024 = getattr(Config, "US_SOYBEAN_ACRES_2024", None)
    except Exception:
        us_acres_by_year = us_acres_2024 = None

    for year in years_to_report:
        total_prod = 0.0
        total_acres = 0.0
        missing_acres = []
        for (f, y), yield_bu in preds.items():
            if y != year:
                continue
            a = acres.get((f, y))
            if a is None:
                missing_acres.append(f)
                continue
            total_prod += yield_bu * a
            total_acres += a

        if total_acres <= 0:
            if missing_acres and year == years_to_report[0]:
                raise SystemExit(
                    f"All FIPS missing in acres CSV for year {year}. "
                    "Add fips, year, harvested_acres for each county and year."
                )
            print(f"Year {year}: no matching acres, skip")
            continue

        us_yield_estimate = total_prod / total_acres
        us_acres = args.us_acres
        if us_acres is None and us_acres_by_year:
            us_acres = us_acres_by_year.get(int(year))
        if us_acres is None and year == "2024" and us_acres_2024:
            us_acres = us_acres_2024

        print(f"Year: {year}")
        print(f"  Sample: {total_acres:,.0f} harvested acres in predicted counties")
        print(f"  US estimated yield (bu/acre): {us_yield_estimate:.2f}")
        if us_acres is not None and us_acres > 0:
            us_production = us_yield_estimate * us_acres
            print(f"  US total harvested acres (input): {us_acres:,.0f}")
            print(f"  US estimated production (bu): {us_production:,.0f}")
        if len(years_to_report) > 1:
            print()


if __name__ == "__main__":
    main()
