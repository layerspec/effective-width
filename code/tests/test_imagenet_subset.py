"""ImageNet class subsets (A26, plan 9.15, 2026-09-26): --class-ids on the ResNet-18 trainer and on the
ruler's ImageNet training-image loader, so a ten-class ImageNet task can be trained and read."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_imagenet_r18 import main                                  # noqa: E402
from scripts.width_from_ruler import imagenet_train_image_loader             # noqa: E402


def _fake_imagenet(root, classes=4, per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for c in range(classes):
            d = os.path.join(root, split, f"n{c:04d}"); os.makedirs(d)
            for i in range(per_class):
                Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(os.path.join(d, f"{i}.jpg"))


def test_ruler_loader_keeps_only_the_listed_classes(tmp_path):
    data = str(tmp_path / "imagenet"); _fake_imagenet(data)
    loader = imagenet_train_image_loader(os.path.join(data, "train"), 100, 4, 0, 0, class_ids=[0, 2])
    ys = torch.cat([y for _, y in loader]).tolist()
    assert len(ys) == 6 and set(ys) == {0, 2}


def test_trainer_trains_a_class_subset(tmp_path):
    data = str(tmp_path / "imagenet"); _fake_imagenet(data)
    out = str(tmp_path / "run")
    assert main(["--data", data, "--out", out, "--device", "cpu", "--workers", "0", "--batch-size", "3", "--crop", "32",
                 "--epochs", "1", "--max-val-batches", "2", "--seed", "0", "--class-ids", "0", "2"]) == 0
    r = json.load(open(os.path.join(out, "result.json")))
    assert r["class_ids"] == [0, 2] and r["num_classes"] == 2 and r["n_train"] == 6
    sd = torch.load(os.path.join(out, "final.pt"), map_location="cpu")
    assert sd["fc.weight"].shape[0] == 2
