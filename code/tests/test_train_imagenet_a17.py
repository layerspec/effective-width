"""CPU checks for scripts/train_imagenet_a17.py (A17 reduced, submission bar D2, 2026-09-25): the
controlled block-type experiment on ImageNet-1k -- basic [6,6,6,6] vs bottleneck ResNet-50, one
recipe, profiles measured on the 6,400-image validation subset along the trajectory."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_imagenet_a17 import build, main   # noqa: E402


def _fake_imagenet(root, classes=3, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for c in range(classes):
            d = os.path.join(root, split, f"n{c:04d}"); os.makedirs(d)
            for i in range(per_class):
                Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(os.path.join(d, f"{i}.jpg"))


def _fake_flat(root, n=8, seed=1):
    rng = np.random.default_rng(seed); os.makedirs(root)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(os.path.join(root, f"ILSVRC_{i:03d}.JPEG"))


def test_builders_have_the_registered_conv_counts_and_imagenet_stem():
    b, r = build("basic6666", 1000), build("resnet50", 1000)
    n = lambda m: sum(isinstance(q, nn.Conv2d) for q in m.modules())   # noqa: E731
    assert n(b) == 52 and n(r) == 53                                 # plan 9.1 / 9.4: 52 vs 53
    for m in (b, r):
        assert m.conv1.kernel_size == (7, 7) and m.conv1.stride == (2, 2) and m.fc.out_features == 1000
    assert build("basic6666", 3).fc.out_features == 3
    r18, r34 = build("resnet18", 1000), build("resnet34", 1000)          # A29b (plan 9.18): the depth axis 20 / 36 / 52
    assert n(r18) == 20 and n(r34) == 36
    assert [len(getattr(r34, f"layer{i}")) for i in (1, 2, 3, 4)] == [3, 4, 6, 3]
    assert all(float(m.bn2.weight.min()) == 1.0 for m in r18.layer1)      # torchvision default: last BN weight at one (gamma = 1)


def test_smoke_trains_measures_and_resumes(tmp_path):
    data, flat, out = str(tmp_path / "imagenet"), str(tmp_path / "val6400"), str(tmp_path / "run")
    _fake_imagenet(data); _fake_flat(flat)
    common = ["--arch", "basic6666", "--data", data, "--measure-data", flat, "--out", out, "--device", "cpu",
              "--workers", "0", "--batch-size", "4", "--crop", "32", "--measure-size", "64", "--positions", "4",
              "--max-train-batches", "2", "--max-val-batches", "1", "--seed", "0", "--schedule", "0", "1", "2"]
    assert main(common + ["--epochs", "2"]) == 0
    t = pd.read_csv(os.path.join(out, "trajectory_layers.csv"))
    assert sorted(t.epoch.unique()) == [0, 1, 2]
    conv = t[(t.kind == "conv") & (t.epoch == 2)]
    assert len(conv) == 52 and {"r_max", "n_over_C", "k_star_0.95", "depth_index", "test_acc", "layer"} <= set(t.columns)
    assert os.path.exists(os.path.join(out, "epoch_000_layers.csv")) and os.path.exists(os.path.join(out, "final.pt"))
    r = json.load(open(os.path.join(out, "result.json")))
    assert r["arch"] == "basic6666" and r["epochs"] == 2 and r["recipe"]["schedule"] == "cosine" and r["macs_224"] > 1e9
    assert not os.path.exists(os.path.join(out, "resume.pt"))
    # resume: a resume.pt at epoch 2 makes --epochs 3 train one more epoch and append its measurement
    m = build("basic6666", 3); m.load_state_dict(torch.load(os.path.join(out, "final.pt"), map_location="cpu"))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": torch.amp.GradScaler("cuda", enabled=False).state_dict(), "rng_cpu": torch.get_rng_state(),
                "rng_cuda": None, "epoch": 2, "history": [], "seed": 0}, os.path.join(out, "resume.pt"))
    assert main(common + ["--epochs", "3"]) == 0
    t = pd.read_csv(os.path.join(out, "trajectory_layers.csv"))
    assert sorted(t.epoch.unique()) == [0, 1, 2, 3]
    assert json.load(open(os.path.join(out, "result.json")))["epochs"] == 3
