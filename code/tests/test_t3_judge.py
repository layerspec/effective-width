"""Tests for scripts/t3_judge.py on synthetic frames."""
import numpy as np
import pandas as pd

from scripts.t3_judge import judge_frames


def _frames(nd_err=0.05, wo_err=0.05, above=1.0, ratio=1.0, acc_drop=0.2, singular_last=True):
    frames = {}
    layers = [f"l{i}" for i in range(10)]
    for arch in ("vgg16_bn", "basic52"):
        for seed in (0, 1, 2):
            none = pd.DataFrame(dict(layer=layers, **{"out_0.95": np.linspace(0.3, 0.5, 10), "data_0.999": [1.0] * 9 + [0.5 if singular_last else 1.0]}))
            cf = pd.DataFrame(dict(layer=layers, nd_pred_95=np.linspace(0.5, 0.7, 10), both_pred_95=np.linspace(0.8, 0.9, 10), ortho_95=np.linspace(0.6, 0.7, 10)))
            nd = pd.DataFrame(dict(layer=layers, **{"out_0.95": cf.nd_pred_95 + nd_err}))
            wo_out = cf.both_pred_95 + wo_err
            if above < 1.0:
                k = int(round((1 - above) * 10)); wo_out[:k] = cf.ortho_95[:k] - 0.05
            wo = pd.DataFrame(dict(layer=layers, **{"out_0.95": wo_out}))
            post = pd.DataFrame(dict(conv=layers, ratio_95=[ratio] * 10))
            frames[(arch, seed)] = dict(none=none, cf=cf, nd=nd, wo=wo, post=post, acc=dict(none=93.0, nd=93.0 - acc_drop, wo=93.0 - acc_drop))
    return frames


def test_all_hold():
    v, text = judge_frames(_frames())
    assert v["P9.128"] and v["P9.129'"] and v["P9.130"] and v["P9.131"]
    assert "excluded ['l9']" in text


def test_each_fails():
    assert not judge_frames(_frames(nd_err=0.2))[0]["P9.128"]
    assert not judge_frames(_frames(wo_err=0.2))[0]["P9.129'"]
    assert not judge_frames(_frames(above=0.5))[0]["P9.129'"]
    assert not judge_frames(_frames(ratio=1.3))[0]["P9.130"]
    assert not judge_frames(_frames(acc_drop=1.5))[0]["P9.131"]


def test_incomplete_is_not_a_pass():
    f = _frames()
    for k in f:
        f[k]["nd"] = None; f[k]["wo"] = None; f[k]["post"] = None; f[k]["acc"] = {}
    v, _ = judge_frames(f)
    assert not any(v[k] for k in ("P9.128", "P9.129'", "P9.130", "P9.131"))
