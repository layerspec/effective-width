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
from .hooks import run_probe


def measure_one(
    model_name: str,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model, weights_tag = _models.build(model_name)
    loader, source = _data.build_loader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        image_size=args.image_size,
        limit=args.limit,
        seed=args.seed,
    )

    t0 = time.time()
    probe = run_probe(
        model,
        loader,
        device=args.device,
        positions_per_image=args.positions,
        include_activations=not args.conv_only,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    layer_rows, conv_rows = [], []
    spectra = {}

    for rec in probe.records():
        if rec.acc is None or rec.acc.n < 2:
            continue
        eig = rec.acc.eigenvalues()
        key = f"{rec.name}::{rec.kind}"
        spectra[key] = eig

        row = rec.meta()
        row.update(_metrics.compute(eig, rec.C, rec.acc.n).to_row())
        row["model"] = model_name
        row["weights"] = weights_tag
        row["source"] = source
        row["n_over_C"] = rec.acc.n / rec.C
        # A sample covariance built from n < C samples is rank-deficient by
        # construction, so its k* is an artefact of the sample size and not a
        # property of the network.  Even well above C the small eigenvalues are
        # biased; we mark anything under n/C = 50 as not reportable and let the
        # convergence file settle the rest.
        row["n_over_C_ok"] = bool(rec.acc.n >= args.min_n_over_C * rec.C)

        # k* against the rank the layer could actually attain, not against its
        # nominal channel count.  For most layers r_max == C and the two agree;
        # where they do not, k*/C understates by exactly the factor the
        # architecture imposed.  Both are written so the paper can show what
        # the choice of denominator does.
        if rec.r_max:
            for tau in _metrics.DEFAULT_TAUS:
                k = row.get(f"k_star_{tau}")
                if k is not None:
                    row[f"k_star_rmax_{tau}"] = k / rec.r_max
            row["r_max_over_C"] = rec.r_max / rec.C

        # The globally-pooled estimator, carried alongside so the paper can test
        # whether the literature's disagreement is the pooling rather than the
        # metric.  It is inherently sample-poor -- n is the image count, so a
        # 2048-channel layer over 50k images only reaches n/C ~ 24 -- which is
        # itself worth reporting.
        if rec.acc_pooled is not None and rec.acc_pooled.n >= 2:
            pm = _metrics.compute(
                rec.acc_pooled.eigenvalues(), rec.C, rec.acc_pooled.n
            ).to_row()
            for k, v in pm.items():
                if k not in ("C",):
                    row[f"pooled_{k}"] = v
            row["pooled_n_over_C"] = rec.acc_pooled.n / rec.C
            row["pooled_n_over_C_ok"] = bool(
                rec.acc_pooled.n >= args.min_n_over_C * rec.C
            )

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
    return pd.DataFrame(layer_rows), pd.DataFrame(conv_rows), spectra, meta


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
    p.add_argument("--min-n-over-C", type=float, default=50.0,
                   help="rows with fewer than this many samples per channel are "
                        "flagged n_over_C_ok=False and must not be reported")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        layers, conv, spectra, meta = measure_one(name, args)

        layers.to_csv(os.path.join(args.out, f"{name}_layers.csv"), index=False)
        conv.to_csv(os.path.join(args.out, f"{name}_convergence.csv"), index=False)
        # Raw eigenvalues, so any metric can be recomputed later without paying
        # for another pass over the data.
        np.savez_compressed(os.path.join(args.out, f"{name}_spectra.npz"), **spectra)
        print(f"  {meta['n_layers']} layers, {meta['elapsed_sec']}s", flush=True)

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

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {args.out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
