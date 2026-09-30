"""CPU checks for A31 (analysis-plan 9.26): the nested class / image lists and the wrapper that runs the A17
training loop on one cell with the two measurement sets."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import a31_lists                                                   # noqa: E402
from scripts.train_imagenet_a31 import CELLS, epochs_for, main                  # noqa: E402
from tests.test_train_imagenet_a17 import _fake_flat, _fake_imagenet            # noqa: E402


def test_epochs_give_37000_steps():
    assert {c: v["epochs"] for c, v in CELLS.items()} == {"k10": 925, "k20": 468, "k50": 189, "k100": 95}
    for c, v in CELLS.items():
        steps = v["epochs"] * -(-v["K"] * 1000 // 256)
        assert abs(steps - 37000) <= 150, (c, steps)
    assert epochs_for(10) == 925


def test_lists_are_nested_disjoint_and_balanced():
    counts = np.array([30] * 6 + [8] * 4)                     # ten classes; the last four have too few images
    t = np.repeat(np.arange(10), counts)
    L = a31_lists.build_lists(t, n_classes=4, min_images=20, holdout=5, per_class=10)
    assert sorted(L["order"]) == sorted(set(L["order"])) and all(c < 6 for c in L["order"]) and len(L["order"]) == 4
    assert a31_lists.cell_classes(L, 2) == sorted(L["order"][:2]) and set(a31_lists.cell_classes(L, 2)) < set(a31_lists.cell_classes(L, 4))
    for c in L["order"]:
        h, tr = L["holdout"][str(c)], L["train"][str(c)]
        assert len(h) == 5 and len(tr) == 10 and not set(h) & set(tr) and set(t[h]) == {c} == set(t[tr])
    ti = a31_lists.train_indices(L, 2)
    assert len(ti) == 20 and set(t[ti]) == set(a31_lists.cell_classes(L, 2))
    hi = a31_lists.holdout_indices(L, 3)
    assert len(hi) == 15 and sorted(t[hi[:3]]) == a31_lists.cell_classes(L, 3)       # round-robin prefix is balanced
    assert a31_lists.build_lists(t, n_classes=4, min_images=20, holdout=5, per_class=10) == L
    a31_lists.check(L, t)
    with pytest.raises(SystemExit):
        a31_lists.check(L, t[::-1])
    with pytest.raises(SystemExit):
        a31_lists.build_lists(t, n_classes=8, min_images=20, holdout=5, per_class=10)   # not enough eligible classes
    with pytest.raises(ValueError):
        a31_lists.cell_classes(L, 5)


def test_wrapper_runs_a_cell_with_both_measurement_sets(tmp_path):
    import scripts.train_imagenet_a17 as a17
    from scripts.imagenet_blob import imagenet_split
    data, flat, out = str(tmp_path / "imagenet"), str(tmp_path / "val6400"), str(tmp_path / "run")
    _fake_imagenet(data, classes=6, per_class=6); _fake_flat(flat)
    t = imagenet_split(data, "train").targets
    L = a31_lists.build_lists(t, n_classes=3, min_images=1, holdout=2, per_class=3)
    path = str(tmp_path / "lists.json"); json.dump(L, open(path, "w"))
    before = a17.loaders
    assert main(["--cell", "k10", "--k", "2", "--data", data, "--lists", path, "--standard", flat, "--out", out, "--device", "cpu",
                 "--workers", "0", "--batch-size", "4", "--crop", "64", "--epochs", "1", "--n-measure", "6", "--positions", "4",
                 "--lr", "0.001", "--max-train-batches", "2", "--max-val-batches", "1"]) == 0
    assert a17.loaders is before
    tr = pd.read_csv(os.path.join(out, "trajectory_layers.csv"))
    assert sorted(tr.epoch.unique()) == [0, 1] and len(tr[(tr.kind == "conv") & (tr.epoch == 1)]) == 53
    s = pd.read_csv(os.path.join(out, "secondary_layers.csv"))
    assert sorted(s.epoch.unique()) == [0, 1]
    r, c = json.load(open(os.path.join(out, "result.json"))), json.load(open(os.path.join(out, "a31.json")))
    assert r["arch"] == "resnet50" and r["recipe"]["schedule"] == "cosine"
    assert c["K"] == 2 and c["classes"] == a31_lists.cell_classes(L, 2) and c["n_secondary_images"] == 4
    import torch
    assert torch.load(os.path.join(out, "final.pt"), map_location="cpu")["fc.weight"].shape[0] == 2
