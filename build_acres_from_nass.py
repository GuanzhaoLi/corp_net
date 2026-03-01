"""
Build fips_harvested_acres_example.csv from NASS Data.csv.
Data.csv should have columns: Year, State ANSI, County ANSI, Data Item, Value.
- SOYBEANS - ACRES HARVESTED -> harvested_acres
- SOYBEANS - YIELD, MEASURED IN BU / ACRE -> (optional, for comparison)
Skips "OTHER COUNTIES". FIPS = State ANSI (2 digits) + County ANSI (3 digits).
"""
import csv
import sys

from config import Config


def _parse_value(s):
    if not s:
        return None
    s = str(s).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    data_path = "Data.csv"
    out_path = "fips_harvested_acres_example.csv"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    acres = {}  # (fips, year) -> acres
    yields = {}  # (fips, year) -> yield bu/acre

    with open(data_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("Commodity") != "SOYBEANS":
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
            val = _parse_value(row.get("Value"))
            if val is None:
                continue
            item = (row.get("Data Item") or "").strip()
            if "ACRES HARVESTED" in item:
                acres[(fips, year)] = int(round(val))
            elif "YIELD" in item and "BU / ACRE" in item:
                yields[(fips, year)] = val

    fips_list = getattr(Config, "US_ESTIMATE_FIPS", [])
    years_wanted = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]  # skip 2025

    # Per-FIPS mean (over years), per-year mean (over fips), and global mean for filling missing
    by_fips = {}
    by_year = {}
    for fips in fips_list:
        by_fips[fips] = [acres[(fips, y)] for y in years_wanted if acres.get((fips, y)) is not None]
    for year in years_wanted:
        by_year[year] = [acres[(f, year)] for f in fips_list if acres.get((f, year)) is not None]
    all_vals = [v for v in acres.values() if v is not None and v > 0]
    global_mean = int(round(sum(all_vals) / len(all_vals))) if all_vals else 0

    def fill_missing(fips, year):
        vals = by_fips.get(fips) or []
        if vals:
            return int(round(sum(vals) / len(vals)))
        vals = by_year.get(year) or []
        if vals:
            return int(round(sum(vals) / len(vals)))
        return global_mean

    rows = []
    missing = []
    for fips in fips_list:
        for year in years_wanted:
            a = acres.get((fips, year))
            if a is None or a <= 0:
                missing.append((fips, year))
                a = fill_missing(fips, year)
            rows.append({"fips": fips, "year": year, "harvested_acres": a})

    if missing:
        print(f"Filled {len(missing)} missing (fips, year) with average: {missing[:5]}...")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "year", "harvested_acres"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows) from {data_path}")

    # Optional: list actual yields for our FIPS (for comparison with predictions)
    if yields:
        out_yields = out_path.replace(".csv", "_actual_yield.csv")
        with open(out_yields, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fips", "year", "actual_yield_bu_per_acre"])
            w.writeheader()
            for r in rows:
                y = yields.get((r["fips"], r["year"]))
                if y is not None:
                    w.writerow({"fips": r["fips"], "year": r["year"], "actual_yield_bu_per_acre": round(y, 2)})
        print(f"Wrote actual yields (where present) to {out_yields}")


if __name__ == "__main__":
    main()
