"""Entry point: measure per-layer effective dimensionality of pretrained CNNs.

    python -m layerspec.run --data /path/to/imagenet/val --out results/

Writes three files per model into --out:
    <model>_layers.csv      one row per layer: metadata + all metrics at full n
    <model>_convergence.csv one row per (layer, n/C) checkpoint
    <model>_spectra.npz     raw eigenvalues per layer, so any metric can be
                            recomputed later without re-running the pass

Nothing here decides anything about the paper's claim -- it produces the
measurement.  Read the convergence file before trusting any row in the layers
file: a layer whose k*/C has not flattened by the largest available n/C is not
a measurement, it is an artefact of sample size (Pospisil & Pillow 2025).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import data as _data
from . import metrics as _metrics
from . import models as _models
from .api import record_row
from .hooks import run_probe


def measure_one(
    model_name: str,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model, weights_tag = _models.build(model_name, seed=args.seed)
    loader, source = _data.build_loader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        image_size=args.image_size,
        limit=args.limit,
        seed=args.seed,
        preprocess=getattr(args, "preprocess", "crop"),
    )

    t0 = time.time()
    probe = run_probe(
        model,
        loader,
        device=args.device,
        positions_per_image=args.positions,
        include_activations=not args.conv_only,
        include_bn=getattr(args, "bn", False),
        max_batches=args.max_batches,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    layer_rows, conv_rows = [], []
    spectra = {}

    for rec in probe.records():
        row = record_row(rec, args.min_n_over_C, _metrics.DEFAULT_TAUS)
        if row is None:
            continue
        spectra[f"{rec.name}::{rec.kind}"] = rec.acc.eigenvalues()
        row["model"] = model_name
        row["weights"] = weights_tag
        row["source"] = source
        layer_rows.append(row)

        for mult, ck in rec.checkpoints.items():
            crow = dict(rec.meta())
            crow.update(ck)
            crow["model"] = model_name
            conv_rows.append(crow)

    meta = {
        "model": model_name,
        "weights": weights_tag,
        "source": source,
        "device": args.device,
        "positions_per_image": args.positions,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "n_layers": len(layer_rows),
        "elapsed_sec": round(elapsed, 1),
        "torch": torch.__version__,
    }
    layers = pd.DataFrame(layer_rows)

    # Sanity check on r_max, per model, before anything is read off the
    # numbers: k*(0.999) cannot exceed the attainable rank.  If it does, the
    # bound is computed wrongly for some layer type in this architecture --
    # which has happened twice (kernel extent forgotten; groups mishandled).
    # Checked over every conv layer, gated or not: the bound is algebraic and
    # does not care about sample size.
    col = "k_star_rmax_0.999"
    if not layers.empty and col in layers.columns:
        conv = layers[(layers.kind == "conv") & layers[col].notna()]
        if not conv.empty:
            worst = conv.loc[conv[col].idxmax()]
            meta["rmax_check_max"] = float(worst[col])
            meta["rmax_check_layer"] = str(worst.layer)
            meta["rmax_check_ok"] = bool(worst[col] <= 1.0 + 1e-9)

    return layers, pd.DataFrame(conv_rows), spectra, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="synthetic",
                   help="ImageNet val directory (root/<class>/<img>), or 'synthetic'")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="+", default=list(_models.DEFAULT_MODELS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--positions", type=int, default=16,
                   help="random spatial positions sampled per image per layer")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on number of images (default: all)")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--conv-only", action="store_true",
                   help="skip activation outputs, hook only Conv2d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bn", action="store_true", help="also record batch-norm outputs (kind=bn; ablation A12)")
    p.add_argument("--preprocess", choices=["crop", "resize"], default="crop",
                   help="centre crop (default) or direct resize (ablation A14)")
    p.add_argument("--min-n-over-C", type=float, default=50.0,
                   help="rows with fewer than this many samples per channel are "
                        "flagged n_over_C_ok=False and must not be reported")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for name in _models.expand_names(args.models):
        print(f"\n=== {name} ===", flush=True)
        layers, conv, spectra, meta = measure_one(name, args)

        stem = _models.file_stem(name)
        layers.to_csv(os.path.join(args.out, f"{stem}_layers.csv"), index=False)
        conv.to_csv(os.path.join(args.out, f"{stem}_convergence.csv"), index=False)
        # Raw eigenvalues, so any metric can be recomputed later without paying
        # for another pass over the data.
        np.savez_compressed(os.path.join(args.out, f"{stem}_spectra.npz"), **spectra)
        print(f"  {meta['n_layers']} layers, {meta['elapsed_sec']}s", flush=True)
        if "rmax_check_ok" in meta:
            flag = "ok" if meta["rmax_check_ok"] else "!! FAILED -- r_max is wrong for this architecture"
            print(f"  r_max check: max k*(0.999)/r_max = {meta['rmax_check_max']:.3f} "
                  f"at {meta['rmax_check_layer']} -> {flag}", flush=True)

        if not layers.empty:
            bad = layers[~layers.n_over_C_ok]
            if not bad.empty:
                worst = bad.loc[bad.n_over_C.idxmin()]
                need = int(np.ceil(
                    args.min_n_over_C * worst.C / max(args.positions, 1)
                ))
                print(
                    f"  !! {len(bad)}/{len(layers)} layers under-sampled "
                    f"(worst: {worst.layer}, C={int(worst.C)}, "
                    f"n/C={worst.n_over_C:.1f}). "
                    f"Need >= {need} images at --positions {args.positions}. "
                    f"These rows are flagged n_over_C_ok=False -- do not report them.",
                    flush=True,
                )
            conv_only = layers[(layers.kind == "conv") & layers.n_over_C_ok]
            if not conv_only.empty:
                r = conv_only["k_star_ratio_0.95"]
                print(f"  k*/C (95% EV, conv layers, adequately sampled): "
                      f"min {r.min():.3f}  max {r.max():.3f}  mean {r.mean():.3f}",
                      flush=True)
        manifest.append(meta)

    write_manifest(args.out, manifest)
    print(f"\nwrote {args.out}/", flush=True)
    return 0


def write_manifest(out: str, entries: list[dict]) -> None:
    """Merge ``entries`` into out/manifest.json, keyed by model name.

    The GPU script runs this module once per stage (pretrained, checkpoint
    tiers, random controls); each stage must add to the manifest rather than
    replace it.  A re-run of the same model replaces its earlier entry.
    """
    path = os.path.join(out, "manifest.json")
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    merged = {e["model"]: e for e in existing}
    for e in entries:
        merged[e["model"]] = e
    with open(path, "w") as f:
        json.dump(list(merged.values()), f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
