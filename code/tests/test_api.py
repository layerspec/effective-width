"""Tests for the public API (layerspec.profile / layerspec.decompose)."""
import numpy as np
import pytest
import torch
import torch.nn as nn


def small_cnn(seed: int = 0) -> nn.Module:
    """Dense 3x3, a 1x1 expansion (r_max binds), a depthwise 3x3, a 1x1 reduction."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 64, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.ReLU(),
        nn.Conv2d(64, 8, 1),
    )


def random_loader(n_images: int = 64, batch: int = 16, size: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_images, 3, size, size, generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, torch.zeros(n_images, dtype=torch.long)),
                                       batch_size=batch, shuffle=False)


def test_profile_table_columns_and_bound():
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(), device="cpu", positions=16)
    t = prof.table
    for col in ["layer", "kind", "depth_index", "C", "groups", "is_depthwise", "r_max",
                "n_samples", "n_over_C", "ok", "k_star_0.95", "k_star_rmax_0.95",
                "k_star_rmax_0.999", "participation_ratio", "effective_rank", "stable_rank"]:
        assert col in t.columns, col
    conv = t[t.kind == "conv"]
    assert len(conv) == 4
    assert (conv["k_star_rmax_0.999"] <= 1.0 + 1e-9).all()
    assert list(conv.r_max) == [16, 16, 64, 8]
    assert list(conv.is_depthwise) == [False, False, True, False]


def test_profile_gate_flags_but_keeps_rows():
    import layerspec
    # 4 images x 4 positions = 16 samples: every layer fails n/C >= 50
    prof = layerspec.profile(small_cnn(), random_loader(n_images=4, batch=4), device="cpu", positions=4)
    conv = prof.table[prof.table.kind == "conv"]
    assert len(conv) == 4
    assert not conv.ok.any()
    assert prof.dense().empty
    # 64 images x 16 positions = 1024 samples: the 16-channel layer passes (n/C = 64), the 64-channel ones do not
    prof = layerspec.profile(small_cnn(), random_loader(), device="cpu", positions=16)
    conv = prof.table[prof.table.kind == "conv"]
    assert conv.ok.tolist() == [True, False, False, True]
    d = prof.dense(tau=0.95)
    assert list(d.columns) == ["layer", "depth_index", "r_max", "k_star_rmax_0.95"]
    assert len(d) == 2                      # dense + ok; the depthwise layer is out even when ok


def test_profile_summary_and_csv_roundtrip(tmp_path):
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(n_images=256), device="cpu", positions=16)
    s = prof.summary(tau=0.95)
    assert set(s) == {"L", "median_level", "rho_depth", "rmax_check_max", "rmax_check_ok"}
    assert s["rmax_check_ok"] is True and s["L"] == 3
    path = tmp_path / "p.csv"
    prof.to_csv(path)
    back = layerspec.Profile.from_csv(path)
    a = prof.table.sort_values(["depth_index", "kind"]).reset_index(drop=True)
    b = back.table.sort_values(["depth_index", "kind"]).reset_index(drop=True)
    assert np.allclose(a["k_star_rmax_0.95"].astype(float), b["k_star_rmax_0.95"].astype(float), equal_nan=True)
    import pandas as pd
    pd.testing.assert_frame_equal(back.dense(), prof.dense(), check_dtype=False)


def test_profile_summary_without_default_taus():
    import layerspec
    # taus does not include 0.999: summary()'s r_max sanity check must not
    # hardcode that column name.
    prof = layerspec.profile(small_cnn(), random_loader(n_images=256), device="cpu",
                             taus=(0.9, 0.95))
    s = prof.summary(tau=0.95)
    assert s["rmax_check_ok"] is True


def test_profile_rejects_models_without_conv():
    import layerspec
    with pytest.raises(ValueError):
        layerspec.profile(nn.Sequential(nn.Flatten(), nn.Linear(192, 4)), random_loader(), device="cpu")


def test_select_device_prefers_available_accelerator():
    from layerspec.api import select_device
    assert select_device("cpu") == "cpu"
    auto = select_device(None)
    assert auto in ("cuda", "mps", "cpu")
