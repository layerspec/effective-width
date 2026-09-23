"""CPU checks for scripts/trajectory_vit.py and scripts/vit_ruler.py (transformer arm, 2026-09-23)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trajectory_vit import build_vit, site_widths, vit_macs, VIT   # noqa: E402
from scripts.vit_ruler import k_star                                        # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")


def test_build_vit_custom_hidden_widths_are_the_sites():
    h = [768, 300, 96, 512, 8, 700]
    m = build_vit(10, h)
    assert site_widths(m) == h
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    full = build_vit(10)
    assert site_widths(full) == [int(VIT["mlp_ratio"] * VIT["embed_dim"])] * VIT["depth"]
    assert vit_macs(m) < vit_macs(full)


def test_k_star_counts_eigenvalues_to_tau():
    eig = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    assert k_star(eig, 0.95) == 3
    assert k_star(eig, 0.999) == 5
    assert k_star(eig, 0.5) == 1


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "cifar-10-batches-py")), reason="CIFAR-10 not downloaded")
def test_controller_smoke_and_resume(tmp_path):
    from scripts.trajectory_vit import main
    out = str(tmp_path / "run")
    common = ["--device", "cpu", "--workers", "0", "--data-root", DATA, "--out", out, "--seed", "0",
              "--max-train-batches", "2", "--max-measure-batches", "2", "--realloc-calib", "256",
              "--realloc-budget", "0.6", "--realloc-epochs", "1", "--schedule", "0", "1", "2"]
    assert main(common + ["--epochs", "2"]) == 0
    # two epochs trained; the controller acted at epoch 1 and left the budget met
    log = json.load(open(os.path.join(out, "realloc.json")))
    assert log[0]["epoch"] == 1 and log[0]["after"] != log[0]["before"]
    assert log[0]["actual_fraction"] <= 0.6 * 1.01
    assert os.path.exists(os.path.join(out, "epoch_002_layers.csv"))
    widths = json.load(open(os.path.join(out, "widths.json")))["widths"]
    assert widths == log[-1]["after"]
    # the resume path rebuilds the model from the recorded widths and loads the weights: shape round-trip
    m = build_vit(10, widths)
    m.load_state_dict(torch.load(os.path.join(out, "vit_c10_cifar10_final.pt"), map_location="cpu"))
    assert site_widths(m) == widths
    # a resume.pt with widths makes the next call resume instead of restart (one more epoch, no new action)
    torch.save({"model": m.state_dict(), "opt": torch.optim.AdamW(m.parameters()).state_dict(), "sched": {},
                "rng_cpu": torch.get_rng_state(), "epoch": 2, "widths": widths}, os.path.join(out, "resume.pt"))
