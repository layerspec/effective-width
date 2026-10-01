"""measure_gradwindow.py (plan 9.27 A/B): window weights from the impulse response, checkpoint reloading, the judge."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_gradwindow import (WINDOW_EPOCHS, build_from_checkpoint, judge_ruler, judge_sweep,   # noqa: E402
                                        measure_run, window_weights)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def test_window_weights_sum_to_one_and_sit_late():
    w, q = window_weights(WINDOW_EPOCHS, 5e-4)
    assert abs(sum(w.values()) - 1) < 1e-9 and set(w) == set(WINDOW_EPOCHS)
    assert sum(v for E, v in w.items() if E >= 60) > 0.85 and 70 <= q["q50"] <= 80 and 0 < q["a_T"] < 1e-3
    q50 = {lam: window_weights(WINDOW_EPOCHS, lam)[1]["q50"] for lam in (5e-5, 5e-4, 5e-3)}
    assert q50[5e-5] <= q50[5e-4] <= q50[5e-3]          # P-b7.4a: a larger decay forgets earlier gradients faster


def test_checkpoint_reloads_with_its_widths(tmp_path):
    from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths
    widths = [8, 8, 12, 12, 16, 16, 16, 20, 20, 20, 10, 10, 10]
    m = build_vgg16_bn_cifar_widths(widths, 10)
    torch.save({k: v for k, v in m.state_dict().items()}, tmp_path / "epoch_100.pt")
    m2, w2 = build_from_checkpoint(str(tmp_path / "epoch_100.pt"), 10)
    assert w2 == widths and torch.equal(m2.features[0].weight, m.features[0].weight)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "cifar-10-batches-py")), reason="CIFAR-10 not downloaded")
def test_measure_run_writes_the_four_widths(tmp_path):
    from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths
    run = tmp_path / "vgg16_bn_ruler_s9"; run.mkdir()
    widths = [8] * 13
    for E in WINDOW_EPOCHS:
        torch.manual_seed(E)
        torch.save(build_vgg16_bn_cifar_widths(widths, 10).state_dict(), run / f"epoch_{E:03d}.pt")
    pd.DataFrame([{"test_acc": 12.5}]).to_csv(run / "epoch_100_layers.csv", index=False)
    df = measure_run(str(run), 5e-4, str(tmp_path / "out"), DATA, "cpu", WINDOW_EPOCHS, n_images=32)
    assert len(df) == 13 and {"k0.95_out", "k0.999_win", "k0.95_end", "k0.999_avg"} <= set(df.columns)
    assert (df["k0.999_win"] <= 8).all() and (df["k0.999_out"] >= 1).all()
    q = json.load(open(tmp_path / "out" / "vgg16_bn_ruler_s9_weights.json"))
    assert q["test_acc"] == 12.5 and q["lam"] == 5e-4 and abs(sum(q["weights"].values()) - 1) < 1e-9
    df2 = measure_run(str(run), 5e-4, str(tmp_path / "out"), DATA, "cpu", WINDOW_EPOCHS, n_images=32)   # cached
    assert df2.equals(df)


def _synthetic(out, name, lam, out_w, win_w, end_w=None, avg_w=None, acc=93.0):
    end_w = win_w if end_w is None else end_w; avg_w = win_w if avg_w is None else avg_w
    C = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    rows = []
    for i in range(13):
        rows.append({"index": i, "layer": f"features.{i}", "C": C[i],
                     "k0.95_out": out_w[i] // 2, "k0.999_out": out_w[i], "k0.95_win": win_w[i] // 2, "k0.999_win": win_w[i],
                     "k0.95_end": end_w[i] // 2, "k0.999_end": end_w[i], "k0.95_avg": avg_w[i] // 2, "k0.999_avg": avg_w[i]})
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{name}_layers.csv"), index=False)
    q50 = {5e-5: 68, 5e-4: 75, 5e-3: 85}[lam]
    json.dump(dict(run=name, lam=lam, weights={"100": 1.0}, q10=q50 - 10, q50=q50, q90=q50 + 8, test_acc=acc),
              open(os.path.join(out, f"{name}_weights.json"), "w"))


def test_judge_ruler(tmp_path, capsys):
    out = str(tmp_path)
    base = [30, 40, 60, 80, 100, 120, 140, 160, 150, 140, 40, 40, 40]
    for s in range(3):
        _synthetic(out, f"vgg16_bn_ruler_s{s}", 5e-4, base, [b + 5 for b in base])
    V = judge_ruler(out)
    assert V == {"P-b7.1'(a)": True, "P-b7.1'(b)": True, "P-b7.1'(c)": True}
    assert "measured inequality" in capsys.readouterr().out
    _synthetic(out, "vgg16_bn_ruler_s2", 5e-4, base, [3 * b for b in base])      # valid but loose
    V = judge_ruler(out)
    assert V["P-b7.1'(b)"] and not V["P-b7.1'(c)"] and "loose" in capsys.readouterr().out


def test_judge_sweep(tmp_path, capsys):
    out = str(tmp_path)
    base = [30, 40, 60, 80, 100, 120, 140, 160, 150, 140, 60, 50, 40]
    for s in (0, 1):
        for lam, shift in ((5e-5, 2), (5e-4, 1), (5e-3, 0)):
            o = [b - 10 * (2 - shift) for b in base]                                 # the late level falls with lambda
            far = [int(b * 1.6) for b in o]
            _synthetic(out, f"wd{lam:g}_s{s}", lam, o, [b + 3 for b in o],
                       end_w=[b + 3 for b in o] if lam == 5e-3 else far, avg_w=[b + 3 for b in o] if lam == 5e-5 else far)
    V = judge_sweep(out)
    assert V == {"P-b7.4b": True, "P-b7.4c": True, "P-b7.4d": True}
    assert "sets the window" in capsys.readouterr().out
