"""A28 (analysis-plan 9.17): the class list and the image lists, fixed before any run.

One generator, numpy.random.default_rng(2026), drawn from in this order:
  1. classes   100 of the 1000 ImageNet classes (labels 0..999 in wnid order), sorted
  2. holdout   for each of the 100 classes in sorted order, 256 of its training images, kept out of training;
               they are the measurement set of the hundred-class cells
  3. n130      for each of the 1000 classes in order, 130 of its training images (the K1000-N130 cell)

The holdout list is stored round-robin over the classes (image 0 of every class, then image 1, ...), so that its
first 6,400 entries are 64 images of each class.  Image lists are indices into the training split as
scripts/imagenet_blob.py orders it, so they are written on the machine that holds the data; the class list
needs no data and is in the repository (scripts/a28_classes.json).

    python scripts/a28_lists.py --classes-only --out scripts/a28_classes.json
    python scripts/a28_lists.py --data /workspace/imagenet --out ../results/a28/lists.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED, N_CLASSES, HOLDOUT, PER_CLASS = 2026, 100, 256, 130


def build_lists(targets=None, n_total: int = 1000, n_classes: int = N_CLASSES, holdout: int = HOLDOUT,
                per_class: int = PER_CLASS) -> dict:
    rng = np.random.default_rng(SEED)
    classes = sorted(int(c) for c in rng.choice(n_total, n_classes, replace=False))
    out = {"seed": SEED, "classes": classes}
    if targets is None:
        return out
    t = np.asarray(targets)
    held = []
    for c in classes:
        idx = np.flatnonzero(t == c)
        if len(idx) <= holdout:
            raise SystemExit(f"class {c} has {len(idx)} training images, cannot hold out {holdout}")
        held.append(np.sort(rng.choice(idx, holdout, replace=False)))
    n130 = []
    for c in range(n_total):
        idx = np.flatnonzero(t == c)
        n130.append(np.sort(rng.choice(idx, min(per_class, len(idx)), replace=False)))
    out.update({"holdout": [int(i) for i in np.stack(held, axis=1).reshape(-1)],     # round-robin over classes
                "n130": [int(i) for i in np.concatenate(n130)],
                "holdout_per_class": holdout, "per_class": per_class, "n_train": int(len(t)),
                "targets_sha1": hashlib.sha1(t.astype(np.int64).tobytes()).hexdigest()})
    return out


def train_indices(lists: dict, targets, cell_classes: bool, per_class: bool) -> np.ndarray:
    """Training indices of a cell: the hundred classes without their held-out images, or 130 images of every class."""
    t = np.asarray(targets)
    if per_class:
        return np.asarray(lists["n130"])
    if not cell_classes:
        return np.arange(len(t))
    keep = np.flatnonzero(np.isin(t, lists["classes"]))
    return np.setdiff1d(keep, np.asarray(lists["holdout"]))


def check(lists: dict, targets) -> None:
    """The lists index one ordering of the training split; refuse any other (the 64-pixel copy must keep it)."""
    t = np.asarray(targets)
    if len(t) != lists["n_train"] or hashlib.sha1(t.astype(np.int64).tobytes()).hexdigest() != lists["targets_sha1"]:
        raise SystemExit("the training split is not the one the A28 lists were drawn from")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=None, help="folder with train/ (scripts/fetch_imagenet.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--classes-only", action="store_true")
    a = p.parse_args(argv)
    if a.classes_only:
        lists = build_lists()
    else:
        from scripts.imagenet_blob import imagenet_split
        lists = build_lists(imagenet_split(a.data, "train").targets)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(lists, fh)
    os.replace(a.out + ".tmp", a.out)
    print({k: (v if not isinstance(v, list) else f"{len(v)} entries") for k, v in lists.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
