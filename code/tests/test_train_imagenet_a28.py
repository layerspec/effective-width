"""CPU checks for A28 (analysis-plan 9.17): the class / image lists, the 80-pixel copy of the blob dataset, and
the wrapper that runs the A17 training loop on one cell."""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import a28_lists                                                   # noqa: E402
from scripts.a28_downsample import main as downsample_main                      # noqa: E402
from scripts.imagenet_blob import BlobImageFolder                               # noqa: E402
from scripts.train_imagenet_a28 import CELLS, main, schedule                    # noqa: E402
from tests.test_train_imagenet_a17 import _fake_flat, _fake_imagenet            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_class_list_is_fixed_and_in_the_repository():
    c = a28_lists.build_lists()["classes"]
    assert len(c) == 100 == len(set(c)) and c == sorted(c) and 0 <= c[0] and c[-1] <= 999
    assert c == a28_lists.build_lists()["classes"]
    assert json.load(open(os.path.join(HERE, "..", "scripts", "a28_classes.json")))["classes"] == c


def test_image_lists():
    t = np.repeat(np.arange(1000), 300)                       # 300 images per class, class-sorted
    L = a28_lists.build_lists(t)
    h = np.asarray(L["holdout"])
    assert len(h) == 100 * 256 == len(set(h)) and set(t[h]) == set(L["classes"])
    assert sorted(t[h[:100]]) == L["classes"]                   # round-robin: the first hundred are one per class
    assert np.bincount(t[h[:6400]], minlength=1000)[L["classes"]].tolist() == [64] * 100
    n130 = np.asarray(L["n130"])
    assert np.bincount(t[n130], minlength=1000).tolist() == [130] * 1000
    tr = a28_lists.train_indices(L, t, cell_classes=True, per_class=False)
    assert len(tr) == 100 * (300 - 256) and not set(tr) & set(h) and set(t[tr]) == set(L["classes"])
    assert len(a28_lists.train_indices(L, t, False, False)) == len(t)
    assert a28_lists.train_indices(L, t, False, True).tolist() == n130.tolist()
    a28_lists.check(L, t)
    with pytest.raises(SystemExit):
        a28_lists.check(L, t[::-1])
    assert a28_lists.build_lists(t)["holdout"] == L["holdout"]


def test_schedule_stretches_with_the_run():
    assert schedule(90) == [0, 1, 2, 5, 10, 20, 30, 45, 60, 75, 90]
    assert schedule(900) == [0, 10, 20, 50, 100, 200, 300, 450, 600, 750, 900]
    assert schedule(2) == [0, 1, 2]
    assert [c for c, v in CELLS.items() if v["crop"] == 64 and v["n_measure"] != 25600] == []   # four positions at stage 4


def _fake_blob(root, shards=2, per_shard=5, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        d = os.path.join(root, split); os.makedirs(d)
        for s in range(shards):
            rows, blob = [], io.BytesIO()
            for i in range(per_shard):
                buf = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (200 + 10 * i, 300, 3), dtype=np.uint8)).save(buf, format="JPEG")
                rows.append((blob.tell(), buf.tell(), (s * per_shard + i) % 3))
                blob.write(buf.getvalue())
            open(os.path.join(d, f"{split}-{s:05d}.blob"), "wb").write(blob.getvalue())
            np.save(os.path.join(d, f"{split}-{s:05d}.idx.npy"), np.asarray(rows, dtype=np.int64))


def test_downsample_keeps_order_and_labels(tmp_path):
    src, dst = str(tmp_path / "in"), str(tmp_path / "in80")
    _fake_blob(src)
    assert downsample_main(["--data", src, "--out", dst, "--workers", "1"]) == 0
    for split in ("train", "val"):
        a, b = BlobImageFolder(os.path.join(src, split)), BlobImageFolder(os.path.join(dst, split))
        assert a.targets == b.targets and len(b) == 10
        for i in (0, 4, 9):
            (x, _), (y, _) = a[i], b[i]
            assert min(y.size) == 80 and abs(y.size[0] / y.size[1] - x.size[0] / x.size[1]) < 0.02
    assert downsample_main(["--data", src, "--out", dst, "--workers", "1"]) == 0   # a finished shard is skipped


def _lists(tmp_path, data, n_classes=3, holdout=2, per_class=3):
    from scripts.imagenet_blob import imagenet_split
    t = imagenet_split(data, "train").targets
    L = a28_lists.build_lists(t, n_total=len(set(t)), n_classes=n_classes, holdout=holdout, per_class=per_class)
    path = str(tmp_path / "lists.json")
    json.dump(L, open(path, "w"))
    return path, L


def test_wrapper_runs_a_hundred_class_cell(tmp_path):
    import scripts.train_imagenet_a17 as a17
    data, flat, out = str(tmp_path / "imagenet"), str(tmp_path / "val6400"), str(tmp_path / "run")
    _fake_imagenet(data, classes=6, per_class=6); _fake_flat(flat)
    path, L = _lists(tmp_path, data)
    before = a17.loaders
    assert main(["--cell", "k100_64", "--data", data, "--lists", path, "--standard", flat, "--out", out, "--device", "cpu",
                 "--workers", "0", "--batch-size", "4", "--crop", "64", "--epochs", "1", "--n-measure", "6", "--positions", "4",
                 "--lr", "0.001", "--max-train-batches", "2", "--max-val-batches", "1"]) == 0   # 64 pixels: stage 4 is 2x2
    assert a17.loaders is before                                        # the patches are undone
    t = pd.read_csv(os.path.join(out, "trajectory_layers.csv"))
    assert sorted(t.epoch.unique()) == [0, 1] and len(t[(t.kind == "conv") & (t.epoch == 1)]) == 53
    s = pd.read_csv(os.path.join(out, "secondary_layers.csv"))
    assert sorted(s.epoch.unique()) == [0, 1]
    r, c = json.load(open(os.path.join(out, "result.json"))), json.load(open(os.path.join(out, "a28.json")))
    assert r["arch"] == "resnet50" and r["recipe"]["schedule"] == "cosine" and r["crop"] == 64
    assert c["cell"] == "k100_64" and c["classes"] == L["classes"] and c["n_measure_images"] == 6
    import torch
    assert torch.load(os.path.join(out, "final.pt"), map_location="cpu")["fc.weight"].shape[0] == 3   # three of six classes


def test_wrapper_runs_the_image_count_cell_without_a_secondary_set(tmp_path):
    data, flat, out = str(tmp_path / "imagenet"), str(tmp_path / "val6400"), str(tmp_path / "run")
    _fake_imagenet(data, classes=6, per_class=6); _fake_flat(flat)
    path, _ = _lists(tmp_path, data)
    assert main(["--cell", "k1000_n130_224", "--data", data, "--lists", path, "--standard", flat, "--out", out, "--device", "cpu",
                 "--workers", "0", "--batch-size", "4", "--crop", "32", "--epochs", "1", "--n-measure", "6", "--positions", "4",
                 "--lr", "0.001", "--max-train-batches", "2", "--max-val-batches", "1"]) == 0
    assert not os.path.exists(os.path.join(out, "secondary_layers.csv"))
    import torch
    assert torch.load(os.path.join(out, "final.pt"), map_location="cpu")["fc.weight"].shape[0] == 6
    assert json.load(open(os.path.join(out, "a28.json")))["n_measure_images"] == 6
