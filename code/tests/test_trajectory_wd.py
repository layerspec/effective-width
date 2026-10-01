"""trajectory_wd.py (plan 9.27 B): the weight decay is replaced, everything else is trajectory_cifar.py."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trajectory_wd import main, sgd_with_decay   # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def test_subclass_forces_the_decay():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = sgd_with_decay(3e-3)([p], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    assert opt.param_groups[0]["weight_decay"] == 3e-3 and opt.param_groups[0]["nesterov"]
    assert torch.optim.SGD.__name__ == "SGD" and torch.optim.SGD is not sgd_with_decay(1.0)


def test_smoke_run_restores_the_optimiser_and_records_the_decay(tmp_path):
    if not os.path.exists(os.path.join(DATA, "cifar-10-batches-py")):
        return
    real = torch.optim.SGD
    out = str(tmp_path / "wd1e-03_s0")
    rc = main(["--wd", "1e-3", "--arch", "vgg16_bn", "--dataset", "cifar10", "--device", "cpu", "--epochs", "1", "--workers", "0",
               "--max-train-batches", "2", "--max-measure-batches", "1", "--schedule", "0", "1", "--save-checkpoints",
               "--data-root", DATA, "--out", out])
    assert rc == 0 and torch.optim.SGD is real
    assert json.load(open(os.path.join(out, "wd.json"))) == {"weight_decay": 1e-3}
    assert os.path.exists(os.path.join(out, "epoch_001.pt")) and os.path.exists(os.path.join(out, "epoch_001_layers.csv"))
