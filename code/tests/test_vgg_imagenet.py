"""A27 (plan 9.16, 2026-09-27): VGG-16 (BN) with the ImageNet head, width-parameterised, trained on an ImageNet
class subset by train_imagenet_r18.py --arch vgg16_bn, and read by width_from_ruler.py --dataset imagenet."""
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.vgg_imagenet import VGG16_WIDTHS, build_vgg16_bn_imagenet, n_params   # noqa: E402
from scripts.train_imagenet_r18 import main as train_main                           # noqa: E402
from scripts.width_from_ruler import main as ruler_main                             # noqa: E402


def _fake_imagenet(root, classes=3, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for c in range(classes):
            d = os.path.join(root, split, f"n{c:04d}"); os.makedirs(d)
            for i in range(per_class):
                Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(os.path.join(d, f"{i}.jpg"))


def test_default_widths_reproduce_torchvision_vgg16_bn():
    m = build_vgg16_bn_imagenet(VGG16_WIDTHS, 10)
    ref = torchvision.models.vgg16_bn(num_classes=10)
    assert n_params(VGG16_WIDTHS, 10) == sum(p.numel() for p in ref.parameters()) == 134309962
    assert set(m.state_dict()) == set(ref.state_dict())
    narrow = build_vgg16_bn_imagenet([32] * 13, 10)
    assert narrow.classifier[0].in_features == 32 * 49 and n_params([32] * 13, 10) < n_params(VGG16_WIDTHS, 10)


def test_trainer_and_ruler_on_the_imagenet_vgg(tmp_path):
    data = str(tmp_path / "imagenet"); _fake_imagenet(data)
    out = str(tmp_path / "run")
    assert train_main(["--arch", "vgg16_bn", "--data", data, "--out", out, "--device", "cpu", "--workers", "0",
                       "--batch-size", "3", "--crop", "64", "--epochs", "1", "--max-val-batches", "2", "--seed", "0",
                       "--class-ids", "0", "2"]) == 0
    r = json.load(open(os.path.join(out, "result.json")))
    assert r["arch"] == "vgg16_bn" and r["num_classes"] == 2 and len(r["widths"]) == 13
    sd = torch.load(os.path.join(out, "final.pt"), map_location="cpu")
    assert sd["classifier.6.weight"].shape[0] == 2
    wj = str(tmp_path / "widths.json")
    assert ruler_main(["--arch", "vgg16_bn", "--dataset", "imagenet", "--tensor", "act", "--checkpoint", os.path.join(out, "final.pt"),
                       "--data", os.path.join(data, "train"), "--class-ids", "0", "2", "--n-images", "8", "--positions", "4",
                       "--workers", "0", "--device", "cpu", "--out", wj, "--no-gate"]) == 0
    w = json.load(open(wj))
    assert len(w["layers"]) == 13 and len(w["widths"]["ruler"]) == 13 and w["params"]["full"] > w["params"]["ruler"] > 0
