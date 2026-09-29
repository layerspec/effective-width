"""checkpoint_analysis.py section for A30 (plan 9.19): two transformer families, P9.51-P9.55."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checkpoint_analysis import A30_FAMILIES, YU_WU_2023_FIG1, section_a30, section_a30_gate   # noqa: E402

ROLES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _write(path, prof, gate=True):
    rows = [{"layer": f"blocks.{b}.{r}", "kind": "linear", "C": 384, "r_max": 384, "n_over_C_ok": gate,
             "k_star_rmax_0.95": prof[r][b]} for b in range(12) for r in ROLES]
    rows.append({"layer": "head", "kind": "linear", "C": 1000, "r_max": 384, "n_over_C_ok": False, "k_star_rmax_0.95": 0.9})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _profile(rng, level, fc2="keep", reorder=False):
    depth = np.linspace(0.0, 0.25, 12)
    base = {r: 0.1 + 0.05 * i + depth for i, r in enumerate(ROLES)}
    out = {}
    for r in ROLES:
        x = base[r][::-1].copy() if (reorder or (r == "mlp.fc2" and fc2 == "scramble")) else base[r].copy()
        if r == "mlp.fc2" and fc2 == "scramble":
            x = rng.permutation(base[r])
        out[r] = x + level + rng.normal(0, 0.003, 12)
    return out


def _families(res, trained_kw, seeds=3, drop=None):
    rng = np.random.default_rng(0)
    for fam, (weights, rand) in A30_FAMILIES.items():
        for w in weights:
            if w != drop:
                _write(os.path.join(res, "a30", f"{w}_layers.csv"), _profile(rng, 0.2, **trained_kw))
        for s in range(1, seeds + 1):
            _write(os.path.join(res, "a30", rand.replace("*", str(s)) + "_layers.csv"), _profile(rng, 0.0))


def test_order_kept_and_level_gained(tmp_path, capsys):
    res = str(tmp_path); _families(res, dict(fc2="scramble"))
    rng = np.random.default_rng(1)
    _write(os.path.join(res, "vit", "vit_base_patch16_224_layers.csv"), _profile(rng, 0.2, fc2="scramble"))
    v = section_a30(res)
    out = capsys.readouterr().out
    assert v["P9.51"] and v["P9.52"] is not None and v["P9.53"] and v["P9.54"] and "P9.55" in v
    assert "48 layers" in out and ("join the families" in out or "no profile shared" in out)


def test_reordered_transformer_is_kept_out(tmp_path, capsys):
    res = str(tmp_path); _families(res, dict(reorder=True))
    v = section_a30(res)
    assert not v["P9.51"] and v["P9.52"] and v["P9.53"]
    assert "not placed with the order-keeping families" in capsys.readouterr().out


def test_incomplete_family_is_not_judged(tmp_path, capsys):
    res = str(tmp_path); _families(res, dict(), drop="deit_base_patch16_224.fb_in1k")
    v = section_a30(res)
    out = capsys.readouterr().out
    assert v == {} and "incomplete" in out and "Registered consequence" not in out
    assert section_a30(str(tmp_path / "none")) is None


def _write_gate(res, shift=0.0, scramble=False):
    rng = np.random.default_rng(2)
    rows = []
    for b in range(12):
        for role, name in (("qkv", "attn.qkv"), ("fc1", "mlp.fc1")):
            ref = list(YU_WU_2023_FIG1[role])
            y = (rng.permutation(ref)[b] if scramble else ref[b]) + shift
            rows.append({"layer": f"blocks.{b}.{name}", "kind": "linear", "C": 2304 if role == "qkv" else 3072, "r_max": 768,
                         "n_over_C_ok": True, "k_star_0.9": round(768 * y)})
    os.makedirs(os.path.join(res, "a30"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(res, "a30", "deit_base_patch16_224.fb_in1k_layers.csv"), index=False)


def test_gate_passes_on_their_values_and_fails_on_a_shift_or_a_scramble(tmp_path):
    a, b, c = (str(tmp_path / x) for x in "abc")
    _write_gate(a); _write_gate(b, shift=0.08); _write_gate(c, scramble=True)
    assert section_a30_gate(a) is True
    assert section_a30_gate(b) is False        # right order, level off by 0.08
    assert section_a30_gate(c) is False        # right level, wrong order
    assert section_a30_gate(str(tmp_path / "none")) is None
