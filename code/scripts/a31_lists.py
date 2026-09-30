"""A31 (analysis-plan 9.26): the class and image lists, fixed before any run.

One generator, numpy.random.default_rng(2031), drawn from in this order:
  1. classes  100 of the ImageNet classes with at least 1,256 training images (911 of the 1000), sorted
  2. order    one permutation of those 100 -- the nesting: K10 is the first ten of `order`, K20 the first twenty, ...
  3. per class, in nesting order: 256 held-out training images (the cell's own measurement set), then 1,000 of
              the remaining ones for training.  Every cell trains on exactly 1,000 images per class.

Image lists are indices into the training split as scripts/imagenet_blob.py orders it, so they are written on
the machine that holds the data (results/a31/lists.json, copied back into the repository).

    python scripts/a31_lists.py --data /workspace/imagenet --out ../results/a31/lists.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED, N_CLASSES, MIN_IMAGES, HOLDOUT, PER_CLASS = 2031, 100, 1256, 256, 1000
CELL_K = {"k10": 10, "k20": 20, "k50": 50, "k100": 100}


def build_lists(targets, n_classes: int = N_CLASSES, min_images: int = MIN_IMAGES, holdout: int = HOLDOUT,
                per_class: int = PER_CLASS) -> dict:
    t = np.asarray(targets)
    counts = np.bincount(t, minlength=int(t.max()) + 1)
    eligible = np.flatnonzero(counts >= min_images)
    if len(eligible) < n_classes:
        raise SystemExit(f"only {len(eligible)} classes have >= {min_images} training images, need {n_classes}")
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(eligible, n_classes, replace=False))
    order = [int(c) for c in chosen[rng.permutation(n_classes)]]
    held, train = {}, {}
    for c in order:
        idx = np.flatnonzero(t == c)
        h = np.sort(rng.choice(idx, holdout, replace=False))
        rest = np.setdiff1d(idx, h)
        if len(rest) < per_class:
            raise SystemExit(f"class {c}: {len(rest)} images left after the hold-out, need {per_class}")
        train[str(c)] = [int(i) for i in np.sort(rng.choice(rest, per_class, replace=False))]
        held[str(c)] = [int(i) for i in h]
    return {"seed": SEED, "order": order, "holdout": held, "train": train, "min_images": min_images,
            "holdout_per_class": holdout, "per_class": per_class, "n_train": int(len(t)),
            "targets_sha1": hashlib.sha1(t.astype(np.int64).tobytes()).hexdigest()}


def cell_classes(lists: dict, k: int) -> list[int]:
    """The first k classes of the nesting order, sorted (the head is renumbered in this order)."""
    if k > len(lists["order"]):
        raise ValueError(f"k = {k} exceeds the {len(lists['order'])} listed classes")
    return sorted(int(c) for c in lists["order"][:k])


def train_indices(lists: dict, k: int) -> np.ndarray:
    return np.sort(np.concatenate([np.asarray(lists["train"][str(c)]) for c in cell_classes(lists, k)]))


def holdout_indices(lists: dict, k: int) -> np.ndarray:
    """Held-out images of the cell's classes, round-robin over the classes so that any prefix is balanced."""
    return np.stack([np.asarray(lists["holdout"][str(c)]) for c in cell_classes(lists, k)], axis=1).reshape(-1)


def check(lists: dict, targets) -> None:
    """The lists index one ordering of the training split; refuse any other."""
    t = np.asarray(targets)
    if len(t) != lists["n_train"] or hashlib.sha1(t.astype(np.int64).tobytes()).hexdigest() != lists["targets_sha1"]:
        raise SystemExit("the training split is not the one the A31 lists were drawn from")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="folder with train/ (scripts/fetch_imagenet.py)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from scripts.imagenet_blob import imagenet_split
    lists = build_lists(imagenet_split(a.data, "train").targets)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(lists, fh)
    os.replace(a.out + ".tmp", a.out)
    print({k: (v if not isinstance(v, (list, dict)) else f"{len(v)} entries") for k, v in lists.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
