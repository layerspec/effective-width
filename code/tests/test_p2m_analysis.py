"""scripts/p2_analysis.py P2m section (ImageNet ResNet-18 controller vs uniform vs A23 full width)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.p2_analysis import p2m_section   # noqa: E402


def _run(root, name, top1, widths, macs_fraction=None, params_fraction=0.6):
    d = os.path.join(root, name); os.makedirs(d)
    json.dump({"final_top1": top1, "final_top5": top1 + 19, "widths": widths, "params_fraction": params_fraction,
               "macs_224": 1.1e9, "epochs": 90, "seed": int(name[-1]), "train_hours": 15.0}, open(os.path.join(d, "result.json"), "w"))
    if macs_fraction is not None:
        json.dump([{"epoch": 10, "shrink_only": False, "before": [64] * 8, "after": widths[2:4] + widths[5:7] + widths[8:10] + widths[11:13],
                    "macs_fraction": 0.93, "macs_before": 1.8e9, "macs_after": 1.7e9},
                   {"epoch": 25, "shrink_only": True, "before": [64] * 8, "after": widths[2:4] + widths[5:7] + widths[8:10] + widths[11:13],
                    "macs_fraction": macs_fraction, "macs_before": 1.7e9, "macs_after": macs_fraction * 1.8e9}],
                  open(os.path.join(d, "realloc.json"), "w"))


def test_p2m_verdicts(tmp_path, capsys):
    res = str(tmp_path)
    full = os.path.join(res, "a18_imagenet"); p2m = os.path.join(res, "p2m")
    _run(full, "resnet18_full_s0", 69.14, [64] * 13, params_fraction=1.0)
    _run(full, "resnet18_full_s1", 68.54, [64] * 13, params_fraction=1.0)
    ctrl = [64, 64, 20, 22, 128, 60, 62, 256, 150, 160, 512, 400, 420]
    _run(p2m, "r18_f60_s0", 68.40, ctrl, macs_fraction=0.599)
    _run(p2m, "r18_f60_s1", 68.10, [w + (2 if i > 1 else 0) for i, w in enumerate(ctrl)], macs_fraction=0.600)
    _run(p2m, "r18_ufixed60_s0", 67.60, [64, 64, 36, 36, 128, 73, 73, 256, 145, 145, 512, 290, 290])
    _run(p2m, "r18_ufixed60_s1", 67.30, [64, 64, 36, 36, 128, 73, 73, 256, 145, 145, 512, 290, 290])
    p2m_section(res)
    out = capsys.readouterr().out
    assert "P2m" in out and "P-m1" in out and "P-m2" in out and "P-m3" in out and "P-m4" in out
    assert "P-m1 (f60 >= ufixed60 + 0.5" in out and "-> yes" in out.split("P-m1")[1].split("\n")[0]
    assert "P-m2" in out and "-> yes" in out.split("P-m2")[1].split("\n")[0]        # 68.25 >= 68.84 - 1.0
    assert "training compute" in out                                                 # 10/90 + 15/90 x 0.93 + 65/90 x 0.60
