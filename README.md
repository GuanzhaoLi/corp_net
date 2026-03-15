# Corp Net — Soybean Yield Prediction

This repo contains models and scripts for predicting soybean yield from satellite (Sentinel-2) imagery. One path uses a spatio-temporal vision model on H5 time-series; another uses a **vision–language model (Qwen2-VL) as a frozen visual encoder plus a regression head** on standalone H5 data. This README focuses on the latter.

---

## Qwen2-VL + Regression Head

We use **Qwen2-VL-2B**’s vision encoder as a frozen feature extractor, then train a small regression head (and optionally a temporal encoder) to predict yield from **5 key frames** per (FIPS, year). No language generation is used—only the visual backbone and a single scalar output (bu/acre).

### Architecture

- **Backbone**: [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) vision encoder, **frozen**.
- **Input**: 5 images per sample (224×224, 3 channels), subsampled from the full time series (e.g. 24 frames) at indices `[0, 6, 12, 18, 23]` to keep memory and sequence length manageable.
- **Temporal option** (`--temporal`): a 1-layer Transformer over the 5 frame-wise features before pooling, so the model can learn temporal patterns instead of collapsing to the mean.
- **Head**: Linear(1280 → 256) → ReLU → Dropout → Linear(256 → 1) → yield (bu/acre).

Checkpoints save only the regression head (and temporal encoder if used), not the 2B-parameter VLM.

### Dependencies

```bash
pip install "transformers>=4.45" accelerate torch torchvision
```

- First run will download Qwen2-VL-2B from Hugging Face (~5GB+ disk).
- GPU: **~10GB+ VRAM** recommended (e.g. batch size 2). Use `--batch-size 1` if needed.

### Data

Same as the standalone pipeline: a directory (e.g. `./standalone_data`) with:

- **`yields.csv`**: columns include `fips`, `year`, and one of `actual_yield_bu_per_acre` / `yield_bu_per_acre` / `predicted_yield_bu_per_acre`.
- **`images/{fips}_{year}.h5`**: each H5 holds time-series images of shape `(T, 3, 224, 224)` (e.g. T=24). The dataloader subsamples **5 frames** per sample (default indices 0, 6, 12, 18, 23).

### Train

```bash
python train_vlm_yield.py \
  --data-dir ./standalone_data \
  --checkpoint-dir ./checkpoints_vlm \
  --epochs 20 \
  --batch-size 2 \
  --normalize-target \
  --temporal
```

| Flag | Meaning |
|------|--------|
| `--normalize-target` | Z-score normalize yield; saves `yield_norm.json` in the checkpoint dir for inverse transform at inference. |
| `--temporal` | Add the 1-layer Transformer over 5 frames (recommended to reduce mean collapse). |
| `--epochs` / `--batch-size` | Training length and batch size (keep batch small for VLM). |

Only the regression head (and temporal encoder when `--temporal` is set) is trained; the vision encoder stays frozen.

### Predict

```bash
python predict_vlm_yield.py \
  --data-dir ./standalone_data \
  --checkpoint ./checkpoints_vlm/vlm_head_best.pth \
  --checkpoint-dir ./checkpoints_vlm \
  --normalize-target \
  --out predictions_vlm.csv
```

Use `--checkpoint-dir` when you used `--normalize-target` so the script can load `yield_norm.json`.

### Files

| File | Role |
|------|------|
| `dataset_vlm_yield.py` | Loads standalone H5, subsamples 5 frames per (fips, year), returns `(5, 3, 224, 224)` + yield. |
| `model_vlm_yield.py` | Builds Qwen2-VL-2B (frozen), optional temporal encoder, mean pool over 5 frames, regression head. |
| `train_vlm_yield.py` | Trains head (and temporal encoder); saves `vlm_head_best.pth`, `vlm_head_last.pth`, and optionally `yield_norm.json`. |
| `predict_vlm_yield.py` | Loads head + VLM, runs inference on the dataset, writes CSV. |

If you see errors related to `pixel_values` / `image_grid_thw` or vision input shape, try upgrading: `pip install -U transformers`.

---

For the main ST-ViT / standalone training and US-level aggregation, see the rest of the repo and `README_US_ESTIMATE.md` (and `README_VLM_YIELD.md` for a Chinese version of this VLM section).
