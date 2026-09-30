"""measure_init_axis.py (plan 9.23): the three initialisations are the ones registered, and the run is resumable."""
import math
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import measure_init_axis as mia   # noqa: E402


def test_jobs_are_the_registered_ones(tmp_path):
    j = mia.jobs(str(tmp_path))
    assert len(j) == 15 and {x["seed"] for x in j} == {1, 2, 3}
    assert sorted({(x["kind"], x["arch"]) for x in j}) == [
        ("gamma0", "resnet50"), ("jax", "vit_base_patch16_224"), ("jax", "vit_small_patch16_224"),
        ("layerscale", "deit3_base_patch16_224"), ("layerscale", "deit3_small_patch16_224")]
    assert len({x["path"] for x in j}) == 15
    assert all(x["transformer"] == (x["kind"] != "gamma0") for x in j)


def test_gamma0_closes_every_residual_branch():
    m = mia.build_model("resnet50", dict(zero_init_last=True), 1)
    last = [p for n, p in m.named_parameters() if n.endswith("bn3.weight")]
    assert len(last) == 16 and all(float(p.detach().abs().max()) == 0.0 for p in last)
    other = [p for n, p in m.named_parameters() if n.endswith("bn1.weight") or n.endswith("bn2.weight")]
    assert all(float(p.detach().min()) == 1.0 for p in other)


def test_jax_is_xavier_uniform_with_small_mlp_biases():
    m = mia.build_model("vit_small_patch16_224", dict(weight_init="jax"), 1)
    w, b = m.blocks[3].mlp.fc1.weight.detach(), m.blocks[3].mlp.fc1.bias.detach()
    bound = math.sqrt(6 / (384 + 1536))
    assert float(w.abs().max()) <= bound and float(w.abs().max()) > 0.99 * bound        # uniform up to the bound
    assert abs(float(w.std()) - bound / math.sqrt(3)) < 1e-3
    assert 0 < float(b.std()) < 1e-5
    assert float(m.blocks[3].attn.qkv.bias.detach().abs().max()) == 0.0
    d = mia.build_model("vit_small_patch16_224", {}, 1)                                 # the A30 random networks
    assert abs(float(d.blocks[3].mlp.fc1.weight.detach().std()) - 0.02) < 1e-3


def test_layerscale_is_the_value_of_the_deit3_code_not_timms():
    for arch in ("deit3_small_patch16_224", "deit3_base_patch16_224"):
        m = mia.build_model(arch, dict(init_values=mia.LAYERSCALE_INIT), 2)
        g = [p.detach() for n, p in m.named_parameters() if n.endswith("ls1.gamma") or n.endswith("ls2.gamma")]
        assert len(g) == 24 and all(torch.allclose(x, torch.full_like(x, 1e-4)) for x in g)
    t = mia.build_model("deit3_small_patch16_224", {}, 2)
    assert abs(float(t.blocks[0].ls1.gamma.detach().mean()) - 1e-6) < 1e-9             # what A30 measured


def test_seed_fixes_the_draw():
    a = mia.build_model("vit_small_patch16_224", dict(weight_init="jax"), 1).blocks[0].attn.qkv.weight.detach()
    b = mia.build_model("vit_small_patch16_224", dict(weight_init="jax"), 1).blocks[0].attn.qkv.weight.detach()
    c = mia.build_model("vit_small_patch16_224", dict(weight_init="jax"), 2).blocks[0].attn.qkv.weight.detach()
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_finished_jobs_are_skipped(tmp_path, monkeypatch, capsys):
    out = str(tmp_path)
    for j in mia.jobs(out):
        os.makedirs(os.path.dirname(j["path"]), exist_ok=True)
        pd.DataFrame({"layer": ["x"]}).to_csv(j["path"] + "_layers.csv", index=False)
    monkeypatch.setattr(mia, "measure", lambda job, a: (_ for _ in ()).throw(AssertionError("measured a finished job")))
    assert mia.main(["--out", out]) == 0
    assert capsys.readouterr().out.count("done, skipping") == 15
