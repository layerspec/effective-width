"""CPU checks for the width controller on the ImageNet ResNet-18 trainer (P2m, 2026-09-24):
scripts/realloc_controller.py (the action shared with the CIFAR trainers) and the
--realloc-* flags of scripts/train_imagenet_r18.py, on a fake three-class ImageFolder."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.realloc import discover_sites                                    # noqa: E402
from scripts.realloc_controller import model_macs, module_hw, reallocate       # noqa: E402
from scripts.resnet_widths import RESNET18_WIDTHS, build_resnet18_widths        # noqa: E402
from scripts.train_imagenet_r18 import main, widths13_from_model                # noqa: E402


def _calib(n=16, hw=64):
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3, hw, hw), torch.zeros(n, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=8)


def _fake_imagenet(root, classes=3, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for c in range(classes):
            d = os.path.join(root, split, f"n{c:04d}")
            os.makedirs(d)
            for i in range(per_class):
                Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(os.path.join(d, f"{i}.jpg"))


def test_reallocate_meets_a_flops_budget_and_keeps_the_residual_stream():
    torch.manual_seed(0)
    model = build_resnet18_widths(list(RESNET18_WIDTHS), 3, "imagenet")
    full = build_resnet18_widths(list(RESNET18_WIDTHS), 3, "imagenet")
    full_macs = model_macs(full, module_hw(full, torch.zeros(1, 3, 64, 64)))
    full_params = sum(q.numel() for q in full.parameters())
    rec = reallocate(model, discover_sites(model), _calib(), "cpu", budget_fraction=0.6, cost="flops",
                     full_params=full_params, full_macs=full_macs, input_hw=(64, 64), min_width=8)
    assert rec["macs_fraction"] <= 0.6 * 1.005 and rec["macs_after"] < rec["macs_before"]
    assert rec["after"] != rec["before"] and min(rec["after"]) >= 8
    assert [model.layer1[0].conv2.out_channels, model.layer4[1].conv2.out_channels] == [64, 512]
    assert model.conv1.out_channels == 64
    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 3, 64, 64)).shape == (1, 3)


def test_widths13_round_trip_after_reallocation():
    torch.manual_seed(0)
    model = build_resnet18_widths(list(RESNET18_WIDTHS), 3, "imagenet")
    assert widths13_from_model(model) == list(RESNET18_WIDTHS)
    full_macs = model_macs(model, module_hw(model, torch.zeros(1, 3, 64, 64)))
    reallocate(model, discover_sites(model), _calib(), "cpu", budget_fraction=0.6, cost="flops",
               full_params=sum(q.numel() for q in model.parameters()), full_macs=full_macs, input_hw=(64, 64))
    w = widths13_from_model(model)
    assert w != list(RESNET18_WIDTHS) and w[0] == 64 and w[1] == 64 and w[10] == 512
    m2 = build_resnet18_widths(w, 3, "imagenet")
    m2.load_state_dict(model.state_dict())                  # shapes match: the 13-vector describes the model
    assert [s.width(m2) for s in discover_sites(m2)] == [s.width(model) for s in discover_sites(model)]


def test_trainer_controller_smoke_and_resume(tmp_path):
    data = str(tmp_path / "imagenet"); _fake_imagenet(data)
    out = str(tmp_path / "run")
    common = ["--data", data, "--out", out, "--device", "cpu", "--workers", "0", "--batch-size", "4",
              "--crop", "32", "--max-train-batches", "2", "--max-val-batches", "1", "--seed", "0",
              "--realloc-budget", "0.6", "--realloc-cost", "flops", "--realloc-epochs", "1", "--realloc-calib", "8"]
    assert main(common + ["--epochs", "2"]) == 0
    log = json.load(open(os.path.join(out, "realloc.json")))
    assert len(log) == 1 and log[0]["epoch"] == 1 and log[0]["after"] != log[0]["before"]
    assert log[0]["macs_fraction"] <= 0.6 * 1.01 and log[0]["cost"] == "flops"
    res = json.load(open(os.path.join(out, "result.json")))
    assert res["epochs"] == 2 and res["widths"] != list(RESNET18_WIDTHS)
    m = build_resnet18_widths(res["widths"], 3, "imagenet")
    m.load_state_dict(torch.load(os.path.join(out, "final.pt"), map_location="cpu"))
    assert [s.width(m) for s in discover_sites(m)] == log[0]["after"]
    assert res["crop"] == 32 and res["macs_224"] > 0 and res["params"] < 11_000_000
    # a resume.pt carrying the reallocated widths makes the next call rebuild that model and continue
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60])
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": torch.amp.GradScaler("cuda", enabled=False).state_dict(), "rng_cpu": torch.get_rng_state(),
                "rng_cuda": None, "epoch": 2, "history": [], "widths": res["widths"], "seed": 0},
               os.path.join(out, "resume.pt"))
    assert main(common + ["--epochs", "3"]) == 0
    res3 = json.load(open(os.path.join(out, "result.json")))
    assert res3["epochs"] == 3 and res3["widths"] == res["widths"]
    assert len(json.load(open(os.path.join(out, "realloc.json")))) == 1     # no new action after resume
