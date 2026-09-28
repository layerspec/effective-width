"""scripts/a27_judge.py: the three A27 predictions (analysis-plan 9.16) read from the ruler's json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.a27_judge import judge, read_seed, section   # noqa: E402

FULL = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]


def _ruler(root, seed, ratios, params=0.5, gate_fail=()):
    os.makedirs(os.path.join(root, "a27"), exist_ok=True)
    ruler = [max(8, int(round(c * r))) for c, r in zip(FULL, ratios)]
    layers = [{"layer": f"features.{i}", "C": c, "ok": i not in gate_fail, "k_star_0.95": w // 2, "k_star_0.999": w}
              for i, (c, w) in enumerate(zip(FULL, ruler))]
    path = os.path.join(root, "a27", f"widths_act_s{seed}.json")
    json.dump({"layers": layers, "widths": {"full": FULL, "ruler": ruler},
               "params": {"full": 1000, "ruler": int(1000 * params)}}, open(path, "w"))
    return path


COLLAPSED = [0.95] * 5 + [0.9] * 5 + [0.5, 0.4, 0.3]
FULL_READ = [0.97] * 13


def test_collapse_in_three_seeds_holds(tmp_path, capsys):
    for s in range(3):
        _ruler(str(tmp_path), s, COLLAPSED, params=0.6)
    v = section(str(tmp_path))
    assert v == {"P9.39": "holds", "P9.40": "holds", "P9.41": "holds"}
    assert "a plain network's late layers collapse" in capsys.readouterr().out


def test_full_width_read_fails_and_names_the_consequence(tmp_path, capsys):
    for s in range(3):
        _ruler(str(tmp_path), s, FULL_READ, params=0.97)
    v = section(str(tmp_path))
    assert v["P9.39"] == "fails" and v["P9.40"] == "fails" and v["P9.41"] == "holds"
    assert "withdrawn" in capsys.readouterr().out


def test_one_seed_out_of_three_breaks_the_prediction(tmp_path):
    runs = [read_seed(_ruler(str(tmp_path), s, COLLAPSED, params=0.6)) for s in range(2)]
    runs.append(read_seed(_ruler(str(tmp_path), 2, [0.95] * 10 + [0.5, 0.4, 0.75], params=0.6)))   # one late layer at 0.75
    assert judge(runs)["P9.39"] == "fails"


def test_fewer_than_three_seeds_is_unjudged_unless_already_broken(tmp_path):
    runs = [read_seed(_ruler(str(tmp_path), 0, COLLAPSED, params=0.6))]
    assert judge(runs) == {"P9.39": "unjudged", "P9.40": "unjudged", "P9.41": "unjudged"}
    runs = [read_seed(_ruler(str(tmp_path), 0, FULL_READ, params=0.97))]
    assert judge(runs)["P9.39"] == "fails"


def test_gated_layer_is_not_reported(tmp_path, capsys):
    for s in range(3):
        _ruler(str(tmp_path), s, COLLAPSED, params=0.6, gate_fail=(12,) if s == 1 else ())
    v = section(str(tmp_path))
    assert v["P9.39"] == "unjudged" and v["P9.41"] == "holds"
    assert "n/a" in capsys.readouterr().out


def test_conv_only_fraction(tmp_path):
    r = read_seed(_ruler(str(tmp_path), 0, [0.5] * 13))
    assert 0.24 < r["conv_params"] < 0.27 and abs(r["params"] - 0.5) < 1e-9   # halving every width quarters the conv stack
