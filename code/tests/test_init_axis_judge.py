"""init_axis_judge.py (plan 9.23, P9.68-P9.71) on synthetic profiles: each verdict can come out both ways."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import init_axis_judge as J   # noqa: E402
from scripts.checkpoint_analysis import A30_FAMILIES   # noqa: E402

L = 20
RNG = np.random.default_rng(0)
BASE0 = RNG.permutation(np.linspace(0.2, 0.6, L))          # the order of the gamma = 0 networks
BASE1 = RNG.permutation(np.linspace(0.2, 0.6, L))          # an unrelated order: the gamma = 1 networks
RESNET_WEIGHTS = {"resnet18": ("tv_in1k", "a1_in1k", "a2_in1k", "gluon_in1k"), "resnet34": ("tv_in1k", "a1_in1k", "a3_in1k"),
                  "resnet101": ("tv_in1k", "tv2_in1k", "a1_in1k"), "wide_resnet50_2": ("tv_in1k", "tv2_in1k", "racm_in1k"),
                  "resnext50_32x4d": ("tv_in1k", "a1_in1k", "a3_in1k")}


def _conv(path, prof):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"layer": [f"l{i}" for i in range(L)], "kind": "conv", "depth_index": range(L), "is_depthwise": False,
                  "n_over_C_ok": True, "r_max": 100, "k_star_0.95": np.round(100 * prof, 3)}).to_csv(path, index=False)


def _noisy(base, s=0.004):
    return base + RNG.normal(0, s, len(base))


def _resnets(res, trained_like):
    """trained_like(recipe group) -> the profile the trained weights of that group follow."""
    grp = lambda tag: "torchvision" if tag in J.TORCHVISION_RECIPES else "timm" if tag in J.TIMM_RECIPES else "other"   # noqa: E731
    for fam, tags in RESNET_WEIGHTS.items():
        for t in tags:
            _conv(os.path.join(res, "local6400", "archs", "trained", f"timm_{fam}.{t}_layers.csv"), _noisy(trained_like(grp(t))))
        for s in (1, 2, 3):
            _conv(os.path.join(res, "local6400", "archs", f"seed{s}", f"timm_{fam}.tv_in1k_random_layers.csv"), _noisy(BASE0))
            _conv(os.path.join(res, "bplus_controls", "gamma1", f"seed{s}", f"timm_{fam}.tv_in1k_random_layers.csv"), _noisy(BASE1))
    for name, tag in (("resnet50", "tv_in1k"), ("timm_resnet50.tv2_in1k", "tv2_in1k"), ("timm_resnet50.a1_in1k", "a1_in1k"),
                      ("timm_resnet50.a3_in1k", "a3_in1k"), ("timm_resnet50.gluon_in1k", "gluon_in1k")):
        _conv(os.path.join(res, "local6400", "trained", f"{name}_layers.csv"), _noisy(trained_like(grp(tag))))
    for s in (1, 2, 3):
        _conv(os.path.join(res, "local6400", f"seed{s}", "resnet50_random_layers.csv"), _noisy(BASE1))
        _conv(os.path.join(res, "init_axis", "gamma0", f"seed{s}", "timm_resnet50.tv_in1k_random_layers.csv"), _noisy(BASE0))


def test_recipe_groups():
    assert J.recipe_group("x/resnet50_layers.csv") == "torchvision"
    assert J.recipe_group("x/timm_resnet50.tv2_in1k_layers.csv") == "torchvision"
    assert J.recipe_group("x/timm_wide_resnet50_2.racm_in1k_layers.csv") == "timm"
    assert J.recipe_group("x/timm_resnet18.gluon_in1k_layers.csv") is None
    assert J.recipe_group("x/timm_resnet18.fb_ssl_yfcc100m_ft_in1k_layers.csv") is None


def test_control_sets_the_number(tmp_path, capsys):
    res = str(tmp_path); _resnets(res, lambda g: BASE0)                 # every weight follows the gamma = 0 order
    v = J.resnets(res)
    assert v == {"P9.68": True, "P9.69": True}
    assert "6/6" in capsys.readouterr().out


def test_match_sets_the_number(tmp_path, capsys):
    res = str(tmp_path); _resnets(res, lambda g: BASE1 if g == "torchvision" else BASE0)   # each group follows its own
    v = J.resnets(res)
    assert v["P9.69"] is False
    assert "the faithful reference matters" in capsys.readouterr().out


def test_incomplete_resnet50_is_not_judged(tmp_path):
    res = str(tmp_path); _resnets(res, lambda g: BASE0)
    for s in (1, 2, 3):
        os.remove(os.path.join(res, "init_axis", "gamma0", f"seed{s}", "timm_resnet50.tv_in1k_random_layers.csv"))
    assert J.resnets(res) == {}


ROLES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _vit(path, prof):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{"layer": f"blocks.{b}.{r}", "kind": "linear", "n_over_C_ok": True, "k_star_rmax_0.95": prof[r][b]}
                  for b in range(12) for r in ROLES]).to_csv(path, index=False)


def _order(seed):
    r = np.random.default_rng(seed)
    return {role: 0.1 + 0.01 * i + r.permutation(np.linspace(0, 0.3, 12)) for i, role in enumerate(ROLES)}


def _shift(o, d):
    return {k: v + d + RNG.normal(0, 0.001, 12) for k, v in o.items()}


def _transformers(res, deit3_like, jax_level, ls_same=True, all_like=None):
    orders = {"default": _order(1), "jax": _order(2), "layerscale": _order(3)}
    for fam, (weights, rand_glob) in A30_FAMILIES.items():
        arch = rand_glob.replace("_random_s*", ""); d3 = arch.replace("vit_", "deit3_")
        for w in weights:
            own = "layerscale" if w.startswith("deit3") else J.FAITHFUL_VIT[w.split(".", 1)[1]]
            like = all_like or (deit3_like if w.startswith("deit3") else own)
            _vit(os.path.join(res, "a30", f"{w}_layers.csv"), _shift(orders[like], 0.3))
        for s in (1, 2, 3):
            _vit(os.path.join(res, "a30", f"{arch}_random_s{s}_layers.csv"), _shift(orders["default"], 0.0))
            _vit(os.path.join(res, "init_axis", "jax", f"{arch}_random_s{s}_layers.csv"), _shift(orders["jax"], jax_level))
            _vit(os.path.join(res, "init_axis", "layerscale", f"{d3}_random_s{s}_layers.csv"), _shift(orders["layerscale"], 0.0))
            if arch.startswith("vit_small"):
                _vit(os.path.join(res, "a30", f"{d3}_random_s{s}_layers.csv"),
                     _shift(orders["layerscale" if ls_same else "jax"], 0.0 if ls_same else 0.05))


def test_transformers_each_weight_follows_its_own_initialisation(tmp_path, capsys):
    res = str(tmp_path); _transformers(res, deit3_like="layerscale", jax_level=-0.05)
    v = J.transformers(res)
    assert v["P9.70a"] and v["P9.71a"] and v["P9.71b"]
    assert v["P9.70b"] is False                      # DeiT-III agrees with its own reference more, not less
    assert "stands under the references" in capsys.readouterr().out


def test_transformers_deit3_follows_the_default_reference(tmp_path, capsys):
    res = str(tmp_path); _transformers(res, deit3_like="default", jax_level=0.0, ls_same=False)
    v = J.transformers(res)
    assert v["P9.70b"] and not v["P9.70a"] and not v["P9.71a"]
    assert v["P9.71b"] is True          # the family median is carried by the weights that follow their own reference


def test_transformers_every_weight_follows_the_default_reference(tmp_path, capsys):
    res = str(tmp_path); _transformers(res, deit3_like="default", jax_level=-0.05, all_like="default")
    v = J.transformers(res)
    assert v["P9.71b"] is False and v["P9.70b"] and v["P9.71a"]
    assert "stated per reference" in capsys.readouterr().out


def test_transformers_incomplete(tmp_path):
    res = str(tmp_path); _transformers(res, deit3_like="layerscale", jax_level=-0.05)
    os.remove(os.path.join(res, "init_axis", "jax", "vit_base_patch16_224_random_s3_layers.csv"))
    assert J.transformers(res) == {}
