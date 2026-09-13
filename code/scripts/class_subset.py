"""Nested class subsets of CIFAR for the class-count control (analysis-plan 9.12, A24).

`classes_for(num_classes, k)` returns the first k labels of one fixed permutation
(`RandomState(CLASS_PERM_SEED)`), so the k=10 subset is contained in k=20, and so on.
`subset_indices` picks the sample indices of those classes, at most `per_class` each
(a seeded choice), and `label_map` renumbers the kept classes 0..k-1 so a k-way head
can be trained.  Additive: with classes=None and per_class=None nothing is filtered.
"""
from __future__ import annotations

import numpy as np

CLASS_PERM_SEED = 0


def classes_for(num_classes: int, k: int | None) -> list[int]:
    if k is None or k >= num_classes:
        return list(range(num_classes))
    if k < 2:
        raise ValueError("need at least two classes")
    perm = np.random.RandomState(CLASS_PERM_SEED).permutation(num_classes)
    return sorted(int(c) for c in perm[:k])


def label_map(classes: list[int]) -> dict[int, int]:
    return {c: i for i, c in enumerate(sorted(classes))}


def subset_indices(targets, classes: list[int], per_class: int | None, seed: int = 0) -> np.ndarray:
    """Indices (sorted) of samples whose label is in `classes`, at most `per_class` per class."""
    t = np.asarray(targets)
    keep = []
    rng = np.random.RandomState(seed)
    for c in sorted(classes):
        idx = np.flatnonzero(t == c)
        if per_class is not None and per_class < len(idx):
            idx = np.sort(rng.choice(idx, per_class, replace=False))
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


class RemapSubset:
    """A dataset view: `base` restricted to `indices`, labels renumbered by `lmap`."""

    def __init__(self, base, indices, lmap: dict[int, int]):
        self.base, self.indices, self.lmap = base, [int(i) for i in indices], lmap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.lmap[int(y)]


def describe(dataset: str, num_classes: int, k: int | None, per_class: int | None, seed: int) -> dict:
    cls = classes_for(num_classes, k)
    return {"dataset": dataset, "classes": cls, "k": len(cls), "per_class": per_class,
            "subset_seed": seed, "class_perm_seed": CLASS_PERM_SEED}
