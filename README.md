# Corp Net — Soybean Price Basis & Yield Prediction

Two parallel models that predict the annual **soybean `price_basis`** (year-over-year log return of the US soybean price) from Sentinel-2 satellite time series plus monthly US macro features. County-level `yield_bu_acre` is kept as an auxiliary label.

- **Route A — Standalone (ST-ViT)**: a hand-built **ViT backbone + Temporal Transformer + MacroEncoder + Gated Fusion** (`model.py::CropPriceModel`).
- **Route B — VLM (Qwen2-VL)**: **Qwen2-VL-2B** vision encoder (frozen) + optional 1-layer temporal encoder + MacroEncoder + Gated Fusion (`model_vlm_yield.py::Qwen2VLPriceModel`).

Both routes share the same dataset layout, the same macro features, and the same train/val split machinery. The only differences are the visual backbone, the fixed 5-frame subsampling in the VLM route, and the checkpoint layout.

> Target note: the primary training target is the annual `price_basis` (log-return of US soybean price). Yield is loaded and normalized for auxiliary use; the heads in the price models regress a single scalar (`price_basis`).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Data](#2-data)
3. [Route A — Standalone (ST-ViT)](#3-route-a--standalone-st-vit)
4. [Route B — VLM (Qwen2-VL frozen)](#4-route-b--vlm-qwen2-vl-frozen)
5. [Monthly national macro injection](#5-monthly-national-macro-injection)
6. [Grid search](#6-grid-search)
7. [Aggregating county predictions to US](#7-aggregating-county-predictions-to-us)
8. [Utilities](#8-utilities)

---

## 1. Repository layout

| File | Role |
|------|------|
| `config.py` | Global `Config` dataclass: model dims, training defaults, FIPS list, `MONTHLY_NATIONAL_MACRO` flag. |
| `model.py` | `CropYieldModel`, `CropPriceModel` (standalone ST-ViT + MacroEncoder + Gated Fusion). |
| `model_vlm_yield.py` | `Qwen2VLYieldModel`, `Qwen2VLPriceModel` (frozen Qwen2-VL-2B + heads + fusion). |
| `dataset_standalone.py` | `StandaloneCropYieldDataset` (all 24 frames per sample, yearly or monthly-aligned macro). |
| `dataset_vlm_yield.py` | `VLMPriceDataset` (5-frame subsample, default indices `[0,6,12,18,23]`). |
| `train_standalone.py` | Train `CropPriceModel`; supports split-by-year, monthly macro, full HP CLI. |
| `train_vlm_price.py` | Train `Qwen2VLPriceModel` (only head / fusion / macro / optional temporal encoder trainable). |
| `predict_standalone.py` | Load ST-ViT checkpoint, run inference, write predictions CSV. |
| `predict_vlm_price.py` | Load VLM checkpoint, run inference, write predictions CSV. |
| `run_grid_search.py` | Hyperparameter sweep driver for both routes; writes summary CSV + loss-curve PNG. |
| `inspect_h5_temporal.py` | Sanity-check H5 frame count, dates, median frame gap. |
| `aggregate_price_basis_to_us.py`, `generalize_to_us.py`, `aggregate_to_us.py` | County → US aggregation helpers. |
| `build_acres_from_nass.py` | Build a `yields.csv` from raw NASS `Data.csv` (county subset, currently hardcoded). |

---

## 2. Data

A typical `standalone_data/` directory:

```
standalone_data/
├── yields.csv          # fips, year, actual_yield_bu_per_acre  (annual)
├── macro_data.csv      # date, crude_oil_usd, usd_index, fed_funds_rate, cpi_yoy,
│                       # soybean_corn_ratio, ...                 (monthly, national)
└── images/{fips}_{year}.h5   # (T=24, 3, 224, 224) Sentinel-2 time series,
                              # 2 frames per month (1st and 15th)
```

Plus in the repo root:

- `us_price_basis.csv` — `year, us_predicted_price_basis, n_counties, method`. Loaded by both datasets to build the `year → price_basis` lookup.
- `Data.csv` — raw NASS download (county-year acres & yields). Used by `generalize_to_us.py` and `build_acres_from_nass.py`.

Current dataset size: **11 counties × 10 years (2016–2025) = 110 candidate samples**; ~65 pass the "has yield + has H5" filter. This is small; see the [Grid search](#6-grid-search) section for the resulting overfitting pattern.

---

## 3. Route A — Standalone (ST-ViT)

### Train

```bash
python train_standalone.py \
  --data-dir ./standalone_data \
  --checkpoint-dir ./checkpoints_standalone \
  --normalize-target \
  --split-by-year --split-seed 42 --val-year-fraction 0.2 \
  --epochs 20
```

All HP flags (new in this version):

| Flag | Default | Meaning |
|------|---------|---------|
| `--lr` | `config.LEARNING_RATE` (5e-5) | Adam LR. |
| `--batch-size` | 4 | Batch size. |
| `--dropout` | 0.1 | Dropout for temporal encoder / macro encoder / head. |
| `--temporal-layers` | 2 | Number of `TransformerEncoderLayer`s over time. |
| `--temporal-heads` | 4 | Attention heads in the temporal encoder. |
| `--weight-decay` | 0.0 | Adam weight decay. |
| `--normalize-target` | off | Z-score normalize `price_basis`; saves `price_basis_norm.json`. |
| `--split-by-year` | off | Val set is whole calendar years (no label leakage across splits). |
| `--val-year-fraction` | 0.2 | Fraction of distinct years held out for val. |
| `--split-seed` | None | If set, shuffles years with this seed before carving out val; also seeds torch/numpy/random. |
| `--monthly-national-macro` | off | Feed per-frame monthly macro `(T, M)` instead of a single yearly `(M,)` vector. |
| `--national-monthly-csv` | `{data-dir}/macro_data.csv` | Override the monthly macro source. |
| `--macro-features` | 5-column default | Subset of macro columns to use. |
| `--crop` | `soybean` | `soybean` or `corn`. |

Outputs under `--checkpoint-dir`:

```
model_best.pth                   # lowest-val-loss weights
model_last.pth                   # final-epoch weights
training_history.json            # train_loss / val_loss per epoch
standalone_train_meta.json       # full HP snapshot + split info (used by predict script)
price_basis_norm.json            # if --normalize-target
```

### Predict

```bash
python predict_standalone.py \
  --data-dir ./standalone_data \
  --checkpoint-dir ./checkpoints_standalone \
  --normalize-target \
  --out predictions_standalone.csv
```

Reads `standalone_train_meta.json` to auto-match yearly vs monthly macro. You can override with `--national-monthly-csv` or force yearly with `--force-yearly-macro`.

---

## 4. Route B — VLM (Qwen2-VL frozen)

Qwen2-VL-2B vision encoder is **frozen**; only the tiny head, macro encoder, gated fusion, and (optional) 1-layer temporal encoder are trainable. First run downloads `Qwen/Qwen2-VL-2B-Instruct` (~5GB).

### Dependencies

```bash
pip install "transformers>=4.45" accelerate torch torchvision
```

VRAM: T4 (15GB) is enough with `--batch-size 2`; use `--batch-size 1` if OOM.

### Train

```bash
python train_vlm_price.py \
  --data-dir ./standalone_data \
  --checkpoint-dir ./checkpoints_vlm_price \
  --epochs 12 --batch-size 2 --lr 5e-4 \
  --normalize-target \
  --temporal \
  --split-by-year --split-seed 42 --val-year-fraction 0.2 \
  [--monthly-national-macro]
```

Outputs under `--checkpoint-dir`:

```
vlm_price_best.pth / vlm_price_last.pth   # only head + fusion + macro (+ temporal + macro_time_embed)
training_history.json
vlm_price_meta.json
price_basis_norm.json
```

The 2B-param VLM is **never** saved; only the trainable modules are in the `.pth`.

### Predict

```bash
python predict_vlm_price.py \
  --data-dir ./standalone_data \
  --checkpoint ./checkpoints_vlm_price/vlm_price_best.pth \
  --normalize-target \
  --out predictions_vlm_price.csv
```

The script reads `vlm_price_meta.json` next to the checkpoint to auto-match monthly vs yearly macro.

---

## 5. Monthly national macro injection

A toggle that exists on both routes: whether macro is supplied as a single yearly pooled vector or as per-frame monthly rows.

- **Yearly (default, `MONTHLY_NATIONAL_MACRO=False`)**: model input is `(B, M)`. Fed into `MacroEncoder` directly, then gated-fused with the temporal-pooled visual context.
- **Monthly (`--monthly-national-macro`)**: model input is `(B, T, M)` where each frame's macro row is looked up by the H5's date metadata. The monthly rows are (a) projected per-frame and **added to the per-frame visual features** before the temporal encoder, and (b) masked-mean pooled and then fed through the `MacroEncoder` for the fusion branch (so the gated-fusion interface is unchanged).

Columns actually varying month-to-month in the current `macro_data.csv`: `crude_oil_usd, usd_index, fed_funds_rate, cpi_yoy, soybean_corn_ratio`. The `yield_bu_acre` column is the national annual figure repeated 12×/year (not a true monthly trajectory).

Column aliasing (e.g. `soy_price` ↔ `soybean_price`) is handled in `dataset_standalone.py`, so `df_raw.csv` and `macro_data.csv` are both usable.

---

## 6. Grid search

`run_grid_search.py` sweeps both routes, collects each run's `training_history.json`, writes a summary CSV/JSON, and produces a loss-curve PNG with one line per combo.

```bash
# Full sweep (default grid: 16 standalone + 4 VLM combos)
python run_grid_search.py --mode both \
  --standalone-epochs 20 --vlm-epochs 12 \
  --split-seed 42 --val-year-fraction 0.2 \
  --out ./grid_search_results --clean

# Smoke test (few combos, few epochs)
python run_grid_search.py --mode standalone --quick --standalone-epochs 3

# Re-plot + re-summarize existing runs without retraining
python run_grid_search.py --mode both --plot-only --out ./grid_search_results
```

Grids are defined at the top of the script and easy to edit. Background-run template:

```bash
nohup python -u run_grid_search.py --mode both \
  --standalone-epochs 20 --vlm-epochs 12 \
  --split-seed 42 --val-year-fraction 0.2 \
  --out ./grid_search_results --clean \
  > ./grid_search_results/sweep.log 2>&1 &
echo $! > ./grid_search_results/sweep.pid
```

Artifacts under `--out`:

```
grid_search_results/
├── loss_curves_standalone.png   # overlay of all standalone runs, ★ = best by val
├── loss_curves_vlm.png          # same for VLM
├── summary_standalone.csv/json  # one row per run: hp + best_val + best_epoch + per-epoch losses
├── summary_vlm.csv/json
├── manifest.json                # sweep args + timestamp
├── sweep.log                    # driver stdout
├── standalone/
│   ├── 01_lr=0.0001_dropout=0.1_temporal_layers=2_monthly_national_macro=F/
│   │   ├── model_best.pth / model_last.pth
│   │   ├── training_history.json / standalone_train_meta.json
│   │   ├── price_basis_norm.json / train.log
│   └── ...
└── vlm/
    └── 01_.../ ...
```

Empirical finding on the current 65-sample dataset: all configurations overfit within 1–2 epochs and no combo beats the "predict the val-set mean" baseline in normalized MSE (see `summary_*.csv`). Monthly macro injection is neutral-to-slightly-worse; reducing model capacity (`temporal_layers=2`, `resnet18` backbone) and increasing weight decay are the most promising regularizers to try next.

---

## 7. Aggregating county predictions to US

Two helpers:

```bash
# Yield-oriented (ST-ViT yield model outputs)
python generalize_to_us.py predictions_standalone.csv Data.csv \
  --method full --out us_estimates.csv

# Price-basis aggregation
python aggregate_price_basis_to_us.py predictions_standalone.csv \
  --out us_price_basis.csv
```

`generalize_to_us.py --method full` uses your predictions for counties you covered and falls back to NASS yields for everything else, then sums `yield × acres` for the US total.

---

## 8. Utilities

| Script | Purpose |
|--------|---------|
| `inspect_h5_temporal.py <path>` | Print `T`, frame dates, median gap. |
| `inspect_h5.py` / `inspect_h5_deep.py` | Shape and per-band inspection of H5 files. |
| `plot_training_history.py` | Plot a single `training_history.json`. |
| `plot_loss.py` | Alternate loss plotting helper. |
| `analyze_price_basis_trends.py` | Per-year / per-county summary of `price_basis`. |
| `eval_vlm_predictions.py` | Compare VLM predictions with ground-truth `price_basis`. |
| `build_acres_from_nass.py` | Extract a `yields.csv` from NASS `Data.csv` (currently hardcoded to 11 counties; needs `--all-counties` for a bigger dataset). |
| `convert_cropnet_to_standalone.py` | One-off converter from the legacy cropnet H5 layout. |
| `verify_dataset.py` | Structural sanity check of `standalone_data/`. |

---

## Also see

- `README_US_ESTIMATE.md` — deeper notes on US aggregation and NASS schema.
- `README_VLM_YIELD.md` — legacy yield-only notes for the VLM route (Chinese).
