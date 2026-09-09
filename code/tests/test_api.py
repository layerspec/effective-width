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
    assert set(s) == {"L", "median_level", "rho_depth", "rmax_check_max", "rmax_check_ok", "rmax_check_tau"}
    assert s["rmax_check_ok"] is True and s["L"] == 3
    assert s["rmax_check_tau"] == 0.999
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
    assert s["rmax_check_tau"] == 0.95    # largest available tau, not a hardcoded 0.999


class _UnusedConv(nn.Module):
    """Holds a Conv2d it never calls in forward -- the hook never fires."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return x


def test_profile_dense_and_summary_on_empty_table():
    import layerspec
    # Conv2d present structurally (so profile() itself does not reject the
    # model) but never invoked in forward: no LayerRecord is ever created.
    prof = layerspec.profile(_UnusedConv(), random_loader(n_images=4, batch=4, size=8), device="cpu")
    assert prof.table.empty
    d = prof.dense()
    assert d.empty
    assert list(d.columns) == ["layer", "depth_index", "r_max", "k_star_rmax_0.95"]
    s = prof.summary()
    assert s["L"] == 0
    assert np.isnan(s["median_level"]) and np.isnan(s["rho_depth"]) and np.isnan(s["rmax_check_max"])

    # Same for an empty loader.
    empty_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(0, 3, 8, 8)),
                                               batch_size=4)
    prof2 = layerspec.profile(small_cnn(), empty_loader, device="cpu")
    assert prof2.table.empty
    assert prof2.dense().empty
    assert prof2.summary()["L"] == 0


def test_profile_dense_and_summary_raise_on_missing_tau():
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(n_images=256), device="cpu",
                             taus=(0.9, 0.95))
    with pytest.raises(ValueError, match="0.99"):
        prof.dense(tau=0.99)
    with pytest.raises(ValueError, match="0.99"):
        prof.summary(tau=0.99)


def test_decompose_raises_when_no_patch_covariance():
    import layerspec
    # 3x3 conv on a 3x3 input: output is 1x1, F.unfold gives L == 1, so
    # PatchProbe never accumulates -- probe.acc stays empty.
    net = nn.Sequential(nn.Conv2d(3, 4, 3))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 3, 3, 3)), batch_size=4)
    with pytest.raises(ValueError, match="no dense convolution accumulated a patch covariance"):
        layerspec.decompose(net, loader, device="cpu")

    empty_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(0, 3, 8, 8)),
                                               batch_size=4)
    with pytest.raises(ValueError, match="no dense convolution accumulated a patch covariance"):
        layerspec.decompose(small_cnn(), empty_loader, device="cpu")


def test_profile_warnings():
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(n_images=4, batch=4), device="cpu", positions=4)
    w = prof.warnings()
    assert len(w) > 0
    assert "under-sampled" in w[0]
    conv = prof.table[prof.table.kind == "conv"]
    worst = conv.loc[conv.n_over_C.idxmin()]
    assert worst.layer in w[0]

    prof2 = layerspec.profile(small_cnn(), random_loader(n_images=256), device="cpu", positions=16)
    assert prof2.warnings() == []


def test_decompose_check_flags_undersampled():
    import layerspec
    dec = layerspec.decompose(small_cnn(), random_loader(n_images=4, batch=4), device="cpu", positions=4)
    c = dec.check()
    assert c["n_below_50"] == len(dec.table) > 0
    assert "n_over_d_min" in c
    assert "n_below_50" not in dec.table.columns          # check_rows()-only, not a new CSV column


def test_profile_and_decompose_restore_training_mode():
    import layerspec
    m = small_cnn()
    m.train()
    layerspec.profile(m, random_loader(n_images=256), device="cpu", positions=16)
    assert m.training is True

    m2 = small_cnn()
    m2.train()
    layerspec.decompose(m2, random_loader(n_images=256), device="cpu", positions=16)
    assert m2.training is True


def test_record_row_matches_paper_csv_header():
    import os
    from layerspec import api, metrics
    from layerspec.hooks import SpectrumProbe

    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "results", "local6400", "trained", "resnet50_layers.csv"))
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")

    net = small_cnn()
    probe = SpectrumProbe(net, positions_per_image=16, pooled=True)
    with torch.no_grad():
        for x, _ in random_loader(n_images=256):
            net(x)
    probe.remove()
    rec = next(r for r in probe.records() if r.kind == "conv")
    row = api.record_row(rec, min_n_over_C=50, taus=metrics.DEFAULT_TAUS)
    assert row is not None
    row_cols = set(row) | {"model", "weights", "source"}

    import pandas as pd
    header = set(pd.read_csv(path, nrows=0).columns)
    assert row_cols == header


def test_profile_rejects_models_without_conv():
    import layerspec
    with pytest.raises(ValueError):
        layerspec.profile(nn.Sequential(nn.Flatten(), nn.Linear(192, 4)), random_loader(), device="cpu")


def test_select_device_prefers_available_accelerator():
    from layerspec.api import select_device
    assert select_device("cpu") == "cpu"
    auto = select_device(None)
    assert auto in ("cuda", "mps", "cpu")


def test_decompose_columns_and_propositions():
    import layerspec
    dec = layerspec.decompose(small_cnn(), random_loader(n_images=256), device="cpu", positions=16)
    t = dec.table
    for col in ["layer", "depth_index", "C_in", "C_out", "k", "d", "r_max", "n_samples", "n_over_d",
                "kernel_erank_over_rmax", "kernel_cond_top_rmax", "data_erank_over_rmax",
                "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95", "ostrowski_ok", "kappa"]:
        assert col in t.columns, col
    assert len(t) == 3                                   # the depthwise layer (groups=64) is not dense
    assert t.ostrowski_ok.all()
    for tau in (0.9, 0.95, 0.99, 0.999):
        assert t[f"bound_ok_{tau}"].all()               # corollary: k*_out(tau) <= k*_ortho(tau')
    c = dec.check()
    assert c["ostrowski"] == (3, 3) and c["kappa_median"] > 0


def test_decompose_identity_matches_direct_measurement():
    """Sigma_out = W Sigma_patch W^T: the spectrum recomputed from the kernel
    and the patch covariance agrees with the directly accumulated output
    spectrum on a 1x1 layer (patch == input pixel), up to sampling noise."""
    import layerspec
    from layerspec.hooks import SpectrumProbe
    torch.manual_seed(1)
    net = nn.Sequential(nn.Conv2d(16, 8, 1))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2000, 16, 8, 8, generator=torch.Generator().manual_seed(3))),
        batch_size=100)
    dec = layerspec.decompose(net, loader, device="cpu", positions=16, seed=0)
    probe = SpectrumProbe(net, positions_per_image=16, pooled=False, include_activations=False, seed=0)
    with torch.no_grad():
        for (x,) in loader:
            net(x)
    rec = [r for r in probe.records() if r.kind == "conv"][0]
    probe.remove()
    lam_direct = np.sort(rec.acc.eigenvalues())[::-1]
    W = net[0].weight.detach().double().reshape(8, 16).numpy()
    Sig = dec.patch_covariance["0"]
    lam_recomputed = np.sort(np.linalg.eigvalsh(W @ Sig @ W.T))[::-1]
    assert np.allclose(lam_direct / lam_direct.sum(), lam_recomputed / lam_recomputed.sum(), atol=0.02)


def test_decompose_rejects_models_without_dense_conv():
    import layerspec
    with pytest.raises(ValueError):
        layerspec.decompose(nn.Sequential(nn.Conv2d(4, 4, 3, groups=4)), random_loader(), device="cpu")


def test_load_model_and_image_loader_synthetic():
    import layerspec
    m = layerspec.load_model("resnet18_random", seed=0)
    assert isinstance(m, nn.Module)
    loader = layerspec.image_loader(None, batch_size=4, limit=8, image_size=32)
    x = next(iter(loader))
    x = x[0] if isinstance(x, (list, tuple)) else x
    assert tuple(x.shape) == (4, 3, 32, 32)


def test_cli_profile_writes_csv(tmp_path):
    from layerspec.cli import main
    out = tmp_path / "prof.csv"
    rc = main(["profile", "--model", "resnet18_random", "--data", "synthetic", "--limit", "8",
               "--image-size", "64", "--batch-size", "4", "--positions", "4", "--device", "cpu",
               "--out", str(out)])
    assert rc == 0 and out.exists()
    import pandas as pd
    t = pd.read_csv(out)
    assert "k_star_rmax_0.95" in t.columns and (t.kind == "conv").sum() == 20


def test_cli_decompose_writes_csv(tmp_path):
    from layerspec.cli import main
    out = tmp_path / "dec.csv"
    rc = main(["decompose", "--model", "resnet18_random", "--data", "synthetic", "--limit", "8",
               "--image-size", "64", "--batch-size", "4", "--positions", "4", "--device", "cpu",
               "--out", str(out)])
    assert rc == 0 and out.exists()
    import pandas as pd
    assert "ortho_0.95" in pd.read_csv(out).columns
