"""apriori_check.py: arbitrary image sets as inputs (2026-09-26, hostile review 09-25 item 3): a registered
dataset NAME gives three input kinds -- NAME (its real images), NAMEmatched (a Gaussian surrogate with
its own cross-spectral density, stored at results/apriori/rgb_csd_NAME.pt) -- and its own reference."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.apriori_check import csd_path_for, make_loader, register_dataset, source_path_for   # noqa: E402


def test_registered_dataset_gives_real_and_matched_inputs(tmp_path):
    d = tmp_path / "imgs"; d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(d / f"{i}.png")
    register_dataset("toy", str(d))
    assert source_path_for("toy") == str(d)
    assert csd_path_for("toymatched").endswith(os.path.join("results", "apriori", "rgb_csd_toy.pt"))
    loader = make_loader("toy", seed=1, n=4, batch_size=2)
    x, _ = next(iter(loader))
    assert tuple(x.shape) == (2, 3, 224, 224)
    assert len(loader.dataset) == 4
