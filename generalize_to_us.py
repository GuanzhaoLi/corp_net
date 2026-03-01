"""
Generalize county-level predictions to US total: use Data.csv (NASS) for acres and optional NASS yield.
Two modes:
  --method sample: Area-weighted yield over your predicted counties only × config.US_SOYBEAN_ACRES_BY_YEAR (same as aggregate_to_us).
  --method full:   For every county in Data.csv with acres: use your prediction if (fips,year) in predictions, else NASS yield; then US production = sum(yield×acres), US yield = production/total_acres.

Usage:
  python generalize_to_us.py pred_check.csv Data.csv --method full --out us_estimates.csv
  python generalize_to_us.py pred_check.csv Data.csv --method sample
"""
import argparse
import csv
import os


def parse_value(s):
    if not s:
        return None
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def load_nass_data(data_csv):
    """Parse Data.csv (NASS). Returns acres[(fips, year)], nass_yield[(fips, year)]."""
    acres = {}
    nass_yield = {}
    with open(data_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("Commodity") or "").strip() != "SOYBEANS":
                continue
            county = (row.get("County") or "").strip()
            if county == "OTHER COUNTIES" or not county:
                continue
            try:
                sa = int(row.get("State ANSI", 0))
                ca = int(row.get("County ANSI", 0))
            except (ValueError, TypeError):
                continue
            fips = f"{sa:02d}{ca:03d}"
            year = (row.get("Year") or "").strip()
            val = parse_value(row.get("Value"))
            if val is None:
                continue
            item = (row.get("Data Item") or "").strip()
            if "ACRES HARVESTED" in item:
                acres[(fips, year)] = val
            elif "YIELD" in item and "BU / ACRE" in item:
                nass_yield[(fips, year)] = val
    return acres, nass_yield


def load_predictions(pred_csv):
    """(fips, year) -> predicted_yield_bu_per_acre"""
    preds = {}
    with open(pred_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fips = (row.get("fips") or "").strip()
            year = (row.get("year") or "").strip()
            col = row.get("predicted_yield_bu_per_acre")
            if fips and year and col:
                try:
                    preds[(fips, year)] = float(col)
                except ValueError:
                    pass
    return preds


def main():
    parser = argparse.ArgumentParser(description="Generalize county predictions to US (using NASS Data.csv)")
    parser.add_argument("predictions_csv", help="e.g. pred_check.csv: fips, year, predicted_yield_bu_per_acre")
    parser.add_argument("nass_data_csv", help="NASS Data.csv: State ANSI, County ANSI, Year, Data Item, Value")
    parser.add_argument("--method", choices=("sample", "full"), default="full",
                        help="sample = area-weighted yield of predicted counties × US acres; full = all counties, pred or NASS yield")
    parser.add_argument("--out", default=None, help="If set, write year, us_yield_bu_per_acre, us_production_bu to CSV")
    parser.add_argument("--year", default=None, help="Single year to report (default: all years in predictions)")
    args = parser.parse_args()

    acres, nass_yield = load_nass_data(args.nass_data_csv)
    preds = load_predictions(args.predictions_csv)
    if not preds:
        raise SystemExit("No rows in predictions CSV.")

    try:
        from config import Config
        us_acres_by_year = getattr(Config, "US_SOYBEAN_ACRES_BY_YEAR", None)
    except Exception:
        us_acres_by_year = None

    years = sorted(set(y for (_, y) in preds))
    if args.year:
        years = [args.year]

    rows_out = []
    for year in years:
        if args.method == "sample":
            total_prod = 0.0
            total_acres = 0.0
            for (f, y), yield_bu in preds.items():
                if y != year:
                    continue
                a = acres.get((f, y))
                if a is None or a <= 0:
                    continue
                total_prod += yield_bu * a
                total_acres += a
            if total_acres <= 0:
                print(f"Year {year}: no acres in NASS for predicted counties, skip")
                continue
            us_yield_est = total_prod / total_acres
            us_acres = (us_acres_by_year or {}).get(int(year)) if us_acres_by_year else None
            if us_acres is None:
                us_acres = (us_acres_by_year or {}).get(2024) or 86e6
            us_production = us_yield_est * us_acres
            print(f"Year {year} (sample): US yield {us_yield_est:.2f} bu/acre, US production {us_production:,.0f} bu (acres {us_acres:,.0f})")
            rows_out.append({"year": year, "us_yield_bu_per_acre": round(us_yield_est, 2), "us_production_bu": round(us_production, 0), "method": "sample"})
        else:
            # full: every county in NASS with acres; use pred if (fips,year) in preds else NASS yield
            total_prod = 0.0
            total_acres = 0.0
            used_pred = 0
            used_nass = 0
            for (fips, y), a in acres.items():
                if y != year or a <= 0:
                    continue
                if (fips, y) in preds:
                    yld = preds[(fips, y)]
                    used_pred += 1
                elif (fips, y) in nass_yield:
                    yld = nass_yield[(fips, y)]
                    used_nass += 1
                else:
                    continue
                total_prod += yld * a
                total_acres += a
            if total_acres <= 0:
                print(f"Year {year}: no counties with both acres and yield, skip")
                continue
            us_yield_est = total_prod / total_acres
            us_production = total_prod  # actual sum over all counties
            us_acres_report = total_acres
            print(f"Year {year} (full): US yield {us_yield_est:.2f} bu/acre, US production {us_production:,.0f} bu (total acres in data {us_acres_report:,.0f}, counties: pred={used_pred} nass={used_nass})")
            rows_out.append({"year": year, "us_yield_bu_per_acre": round(us_yield_est, 2), "us_production_bu": round(us_production, 0), "total_acres": round(us_acres_report, 0), "method": "full"})

    if args.out and rows_out:
        with open(args.out, "w", newline="") as f:
            fieldnames = ["year", "us_yield_bu_per_acre", "us_production_bu", "method"]
            if rows_out and "total_acres" in rows_out[0]:
                fieldnames.insert(3, "total_acres")
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
