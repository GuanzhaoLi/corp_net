"""
Grid-search driver for Standalone (ST-ViT + MacroEncoder + GatedFusion)
and VLM (Qwen2-VL-2B frozen vision + MacroEncoder + GatedFusion) routes.

Each combination is trained by shelling out to train_standalone.py or
train_vlm_price.py with the right flags; its checkpoint dir is unique so
nothing is overwritten. After all combos finish we collect every run's
training_history.json + standalone_train_meta.json / vlm_price_meta.json,
build a summary CSV/JSON, and draw one loss-curve figure per route with
one line per combo (train + val as two subplots).

Usage:
  # full default grids (standalone + vlm)
  python run_grid_search.py --mode both --out ./grid_search_results

  # just standalone, quick smoke (few combos, few epochs)
  python run_grid_search.py --mode standalone --quick

  # just vlm with custom epoch count (careful: Qwen frozen vision is slow on T4)
  python run_grid_search.py --mode vlm --vlm-epochs 10

  # replot existing runs without retraining
  python run_grid_search.py --plot-only --out ./grid_search_results
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------
# Standalone route: ST-ViT + Temporal Transformer + MacroEncoder + GatedFusion.
# Dataset is tiny (11 counties x 10 years = 110 samples), so keep sweep modest.
STANDALONE_GRID_FULL: Dict[str, List[Any]] = {
    "lr": [1e-4, 5e-5],
    "dropout": [0.1, 0.3],
    "temporal_layers": [2, 4],
    "monthly_national_macro": [False, True],
}
STANDALONE_GRID_QUICK: Dict[str, List[Any]] = {
    "lr": [1e-4],
    "dropout": [0.1],
    "temporal_layers": [2],
    "monthly_national_macro": [False, True],
}

# VLM route: frozen Qwen2-VL-2B vision; only small trainable head + macro + fusion
# (+ optional temporal encoder). Most compute is the frozen vision forward, so
# keep the grid small.
VLM_GRID_FULL: Dict[str, List[Any]] = {
    "lr": [5e-4, 1e-3],
    "monthly_national_macro": [False, True],
    "temporal": [True],  # keep temporal on — adds <1% params, clear upside over pure mean
}
VLM_GRID_QUICK: Dict[str, List[Any]] = {
    "lr": [1e-3],
    "monthly_national_macro": [False, True],
    "temporal": [True],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    mode: str
    run_id: str
    run_name: str
    checkpoint_dir: str
    hyperparams: Dict[str, Any]
    train_loss: List[float]
    val_loss: List[float]
    best_val: Optional[float]
    best_epoch: Optional[int]
    final_train: Optional[float]
    final_val: Optional[float]
    elapsed_sec: float
    returncode: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def _slug(combo: Dict[str, Any]) -> str:
    parts = []
    for k, v in combo.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'T' if v else 'F'}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def _short_label(combo: Dict[str, Any]) -> str:
    """Shorter label for plot legends."""
    bits = []
    if "lr" in combo:
        bits.append(f"lr={combo['lr']:g}")
    if "dropout" in combo:
        bits.append(f"drop={combo['dropout']:g}")
    if "temporal_layers" in combo:
        bits.append(f"L={combo['temporal_layers']}")
    if "monthly_national_macro" in combo:
        bits.append("mnth" if combo["monthly_national_macro"] else "yr")
    if "temporal" in combo:
        bits.append("T+" if combo["temporal"] else "T-")
    return " ".join(bits)


def _load_training_history(ckpt_dir: str) -> Tuple[List[float], List[float]]:
    fp = os.path.join(ckpt_dir, "training_history.json")
    if not os.path.isfile(fp):
        return [], []
    with open(fp) as f:
        d = json.load(f)
    return list(d.get("train_loss", [])), list(d.get("val_loss", []))


def _load_meta(ckpt_dir: str, mode: str) -> Dict[str, Any]:
    name = "standalone_train_meta.json" if mode == "standalone" else "vlm_price_meta.json"
    fp = os.path.join(ckpt_dir, name)
    if not os.path.isfile(fp):
        return {}
    with open(fp) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run launcher
# ---------------------------------------------------------------------------
def _standalone_cmd(
    combo: Dict[str, Any], args: argparse.Namespace, ckpt_dir: str
) -> List[str]:
    cmd = [
        sys.executable,
        "train_standalone.py",
        "--data-dir", args.data_dir,
        "--checkpoint-dir", ckpt_dir,
        "--epochs", str(args.standalone_epochs),
        "--crop", args.crop,
        "--normalize-target",
        "--split-by-year",
        "--val-year-fraction", str(args.val_year_fraction),
        "--split-seed", str(args.split_seed),
        "--lr", str(combo["lr"]),
        "--dropout", str(combo["dropout"]),
        "--temporal-layers", str(combo["temporal_layers"]),
        "--temporal-heads", "4",
        "--batch-size", str(args.standalone_batch_size),
    ]
    if combo.get("monthly_national_macro", False):
        cmd.append("--monthly-national-macro")
    return cmd


def _vlm_cmd(combo: Dict[str, Any], args: argparse.Namespace, ckpt_dir: str) -> List[str]:
    cmd = [
        sys.executable,
        "train_vlm_price.py",
        "--data-dir", args.data_dir,
        "--checkpoint-dir", ckpt_dir,
        "--epochs", str(args.vlm_epochs),
        "--batch-size", str(args.vlm_batch_size),
        "--lr", str(combo["lr"]),
        "--crop", args.crop,
        "--normalize-target",
        "--split-by-year",
        "--val-year-fraction", str(args.val_year_fraction),
        "--split-seed", str(args.split_seed),
    ]
    if combo.get("temporal", False):
        cmd.append("--temporal")
    if combo.get("monthly_national_macro", False):
        cmd.append("--monthly-national-macro")
    return cmd


def _run_one(
    mode: str,
    combo: Dict[str, Any],
    args: argparse.Namespace,
    runs_root: str,
    run_idx: int,
    total: int,
) -> RunResult:
    run_name = _slug(combo)
    run_id = f"{run_idx:02d}_{run_name}"
    ckpt_dir = os.path.join(runs_root, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = _standalone_cmd(combo, args, ckpt_dir) if mode == "standalone" else _vlm_cmd(combo, args, ckpt_dir)

    log_path = os.path.join(ckpt_dir, "train.log")
    t0 = time.time()
    print(
        f"\n>>> [{mode}] run {run_idx}/{total} :: {run_name}\n"
        f"    cmd: {' '.join(cmd)}\n"
        f"    log: {log_path}"
    )
    with open(log_path, "w") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    train_loss, val_loss = _load_training_history(ckpt_dir)
    best_val, best_epoch = None, None
    if val_loss:
        best_val = min(val_loss)
        best_epoch = int(val_loss.index(best_val))
    final_train = train_loss[-1] if train_loss else None
    final_val = val_loss[-1] if val_loss else None

    meta = _load_meta(ckpt_dir, mode)
    effective_hp = dict(combo)
    if meta.get("hyperparams"):
        effective_hp = {**effective_hp, **{k: v for k, v in meta["hyperparams"].items()}}

    status = "OK" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
    print(
        f"    [{status}] elapsed={elapsed:.1f}s "
        f"best_val={best_val if best_val is None else f'{best_val:.4f}'} "
        f"(epoch {best_epoch})"
    )

    return RunResult(
        mode=mode,
        run_id=run_id,
        run_name=run_name,
        checkpoint_dir=ckpt_dir,
        hyperparams=effective_hp,
        train_loss=train_loss,
        val_loss=val_loss,
        best_val=best_val,
        best_epoch=best_epoch,
        final_train=final_train,
        final_val=final_val,
        elapsed_sec=elapsed,
        returncode=proc.returncode,
        extra={
            "val_years": meta.get("val_years"),
            "train_years": meta.get("train_years"),
            "split_seed": meta.get("split_seed"),
            "monthly_national_macro": meta.get("monthly_national_macro"),
        },
    )


# ---------------------------------------------------------------------------
# Collection / plotting
# ---------------------------------------------------------------------------
def _collect_existing_runs(mode: str, runs_root: str) -> List[RunResult]:
    results: List[RunResult] = []
    if not os.path.isdir(runs_root):
        return results
    for entry in sorted(os.listdir(runs_root)):
        ckpt_dir = os.path.join(runs_root, entry)
        if not os.path.isdir(ckpt_dir):
            continue
        train_loss, val_loss = _load_training_history(ckpt_dir)
        if not train_loss and not val_loss:
            continue
        meta = _load_meta(ckpt_dir, mode)
        hp = meta.get("hyperparams", {}) or {}
        combo = {
            "lr": hp.get("lr"),
            "dropout": hp.get("dropout"),
            "temporal_layers": hp.get("temporal_layers"),
            "monthly_national_macro": bool(meta.get("monthly_national_macro", False)),
            "temporal": hp.get("use_temporal", None),
        }
        combo = {k: v for k, v in combo.items() if v is not None}
        best_val = min(val_loss) if val_loss else None
        best_epoch = int(val_loss.index(best_val)) if best_val is not None else None
        results.append(
            RunResult(
                mode=mode,
                run_id=entry,
                run_name=entry.split("_", 1)[1] if "_" in entry else entry,
                checkpoint_dir=ckpt_dir,
                hyperparams={**combo, **hp},
                train_loss=train_loss,
                val_loss=val_loss,
                best_val=best_val,
                best_epoch=best_epoch,
                final_train=train_loss[-1] if train_loss else None,
                final_val=val_loss[-1] if val_loss else None,
                elapsed_sec=0.0,
                returncode=0,
                extra={
                    "val_years": meta.get("val_years"),
                    "train_years": meta.get("train_years"),
                },
            )
        )
    return results


def _plot_loss_curves(results: List[RunResult], mode: str, out_path: str) -> None:
    if not results:
        print(f"[plot] no results for {mode}, skip")
        return
    # Keep only successful runs with histories
    results = [r for r in results if r.train_loss and r.val_loss]
    if not results:
        return
    # Sort by best val (NaN last)
    results = sorted(results, key=lambda r: (r.best_val if r.best_val is not None else float("inf")))

    cmap = plt.get_cmap("tab20")
    n = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    ax_tr, ax_val = axes

    for i, r in enumerate(results):
        color = cmap(i % cmap.N)
        label = _short_label(r.hyperparams)
        # Mark best run with solid, thicker line; others thinner + alpha
        is_best = (i == 0)
        lw = 2.4 if is_best else 1.4
        alpha = 1.0 if is_best else 0.75
        label_with_best = f"★ {label} (val={r.best_val:.3f})" if is_best else f"{label} (val={r.best_val:.3f})"
        xs = list(range(1, len(r.train_loss) + 1))
        ax_tr.plot(xs, r.train_loss, color=color, lw=lw, alpha=alpha, label=label_with_best)
        xs_v = list(range(1, len(r.val_loss) + 1))
        ax_val.plot(xs_v, r.val_loss, color=color, lw=lw, alpha=alpha, label=label_with_best)

    for ax, title in ((ax_tr, "Train loss"), (ax_val, "Val loss")):
        ax.set_title(f"{mode.upper()} — {title} ({n} runs)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalized target)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Single legend on the right
    handles, labels = ax_val.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        f"Grid search loss curves — {mode} ({n} configs, sorted by best val)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.80, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _write_summary(results: List[RunResult], mode: str, out_dir: str) -> None:
    if not results:
        return
    rows = []
    for r in results:
        row = {
            "mode": r.mode,
            "run_id": r.run_id,
            "best_val": r.best_val,
            "best_epoch": r.best_epoch,
            "final_train": r.final_train,
            "final_val": r.final_val,
            "elapsed_sec": round(r.elapsed_sec, 1),
            "returncode": r.returncode,
        }
        for k, v in r.hyperparams.items():
            row[f"hp_{k}"] = v
        if r.extra.get("val_years") is not None:
            row["val_years"] = ",".join(map(str, r.extra["val_years"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "best_val" in df.columns:
        df = df.sort_values("best_val", na_position="last")
    csv_path = os.path.join(out_dir, f"summary_{mode}.csv")
    json_path = os.path.join(out_dir, f"summary_{mode}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "run_id": r.run_id,
                    "hyperparams": r.hyperparams,
                    "best_val": r.best_val,
                    "best_epoch": r.best_epoch,
                    "final_train": r.final_train,
                    "final_val": r.final_val,
                    "elapsed_sec": r.elapsed_sec,
                    "returncode": r.returncode,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "extra": r.extra,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"[summary] wrote {csv_path} and {json_path}")
    # Show top 5
    top = df.head(5)
    print(f"\n[{mode}] Top-5 by best val loss:")
    print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standalone", "vlm", "both"], default="both")
    p.add_argument("--out", default="./grid_search_results", help="Output root dir")
    p.add_argument("--quick", action="store_true", help="Use smaller quick grid for smoke testing")
    p.add_argument("--plot-only", action="store_true", help="Skip training; re-read existing runs and re-plot/summarize")
    p.add_argument("--data-dir", default="./standalone_data")
    p.add_argument("--crop", choices=["soybean", "corn"], default="soybean")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-year-fraction", type=float, default=0.2)
    p.add_argument("--standalone-epochs", type=int, default=25)
    p.add_argument("--standalone-batch-size", type=int, default=4)
    p.add_argument("--vlm-epochs", type=int, default=12)
    p.add_argument("--vlm-batch-size", type=int, default=2)
    p.add_argument("--clean", action="store_true", help="Wipe out/<mode>/ before running")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    modes = ["standalone", "vlm"] if args.mode == "both" else [args.mode]

    all_results: Dict[str, List[RunResult]] = {}
    for mode in modes:
        runs_root = os.path.join(args.out, mode)
        if args.plot_only:
            results = _collect_existing_runs(mode, runs_root)
        else:
            if args.clean and os.path.isdir(runs_root):
                print(f"[clean] removing {runs_root}")
                shutil.rmtree(runs_root)
            os.makedirs(runs_root, exist_ok=True)
            if mode == "standalone":
                grid = STANDALONE_GRID_QUICK if args.quick else STANDALONE_GRID_FULL
            else:
                grid = VLM_GRID_QUICK if args.quick else VLM_GRID_FULL
            combos = _expand_grid(grid)
            print(f"\n=== [{mode}] grid: {len(combos)} combos ===")
            for k, v in grid.items():
                print(f"  {k}: {v}")
            results: List[RunResult] = []
            for i, combo in enumerate(combos, 1):
                r = _run_one(mode, combo, args, runs_root, i, len(combos))
                results.append(r)
                # Save incremental summary so we don't lose progress on crash
                _write_summary(results, mode, args.out)

        all_results[mode] = results
        out_fig = os.path.join(args.out, f"loss_curves_{mode}.png")
        _plot_loss_curves(results, mode, out_fig)
        _write_summary(results, mode, args.out)

    # Combined metadata
    manifest = {
        "modes": modes,
        "args": vars(args),
        "counts": {m: len(all_results[m]) for m in modes},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Results under {args.out}/")


if __name__ == "__main__":
    main()
