"""CPU checks for scripts/trajectory_r20wide.py (P2l, 2026-09-25): ResNet-20 with every stage widened by a
multiplier, trained through trajectory_cifar.main with the same controller, without editing that file."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.realloc import discover_sites                       # noqa: E402
from scripts.trajectory_r20wide import build_resnet20_wide, main    # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")


def test_wide_builder_scales_every_stage_and_keeps_nine_sites():
    m = build_resnet20_wide(10, None, 4)
    assert [m.layers[0].conv2.out_channels, m.layers[3].conv2.out_channels, m.layers[8].conv2.out_channels] == [64, 128, 256]
    assert m.conv1.out_channels == 64 and m.fc.in_features == 256
    sites = discover_sites(m)
    assert len(sites) == 9 and [s.width(m) for s in sites] == [64] * 3 + [128] * 3 + [256] * 3
    base = sum(q.numel() for q in build_resnet20_wide(10, None, 1).parameters())
    wide = sum(q.numel() for q in m.parameters())
    assert 14 < wide / base < 16.5                                  # about 16x, minus the shared stem / fc
    inner = [8, 9, 10, 20, 21, 22, 40, 41, 42]
    assert [s.width(build_resnet20_wide(10, inner, 4)) for s in discover_sites(build_resnet20_wide(10, inner, 4))] == inner
    with torch.no_grad():
        assert m.eval()(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_controller_runs_on_the_wide_net_and_final_widths_rebuild(tmp_path):
    out = str(tmp_path / "run")
    assert main(["--width-mult", "4", "--device", "cpu", "--workers", "0", "--data-root", DATA, "--out", out,
                 "--seed", "0", "--epochs", "2", "--max-train-batches", "2", "--max-measure-batches", "2",
                 "--schedule", "0", "1", "2", "--realloc-budget", "0.6", "--realloc-cost", "flops",
                 "--realloc-epochs", "1", "--realloc-calib", "256"]) == 0
    log = json.load(open(os.path.join(out, "realloc.json")))
    assert log[0]["epoch"] == 1 and log[0]["macs_fraction"] <= 0.6 * 1.01 and log[0]["after"] != log[0]["before"]
    assert log[0]["before"] == [64] * 3 + [128] * 3 + [256] * 3
    m = build_resnet20_wide(10, log[-1]["after"], 4)
    m.load_state_dict(torch.load(os.path.join(out, "resnet20_cifar10_final.pt"), map_location="cpu"))
    assert [s.width(m) for s in discover_sites(m)] == log[-1]["after"]
