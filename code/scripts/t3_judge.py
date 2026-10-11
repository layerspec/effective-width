"""Judgement of plan 9.56 (T3 four arms) from the per-run tables; written before the whitening arms were trained.

Inputs (per arch in {vgg16_bn, basic52}, seed in {0,1,2}):
  decompose    results/round3_decompose/a5_<arch>_<arm>_s<seed>_decompose.csv   (decompose_check.py; arms none, oni, nd, wo)
  counterfact  results/t3/counterfactual_<arch>_none_s<seed>.csv                 (t3_counterfactual.py on the none arm)
  postrelu     results/t3/postrelu_a5_<arch>_wo_s<seed>.csv                      (t3_postrelu.py on the wo arm)
  accuracy     results/round3/a5_<arch>_<arm>_s<seed>/trajectory_layers.csv      (test_acc at the last epoch)
Layers with data_0.999 < 1 in the none arm (near-singular Sigma_patch) are excluded from P9.128 / P9.129' and listed.

  P9.128   median_l |out_nd - nd_pred| < 0.10 in >= 2/3 seeds per arch
  P9.129'  median_l |out_wo - both_pred| < 0.10 in >= 2/3 seeds per arch, and out_wo > ortho_none in >= 80 % of layers (pooled per arch)
  P9.130   ratio_95 = k_act / k_bn in [0.90, 1.10] in >= 80 % of the paired layers of the wo arm (pooled over arch and seed)
  P9.131   acc(nd), acc(wo) >= acc(none, same seed) - 1.0 for every run

    python scripts/t3_judge.py --results ../results
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ARCHS = {"vgg16_bn": "vgg16_bn", "basic52": "basic52"}
SEEDS = (0, 1, 2)
TAU = "0.95"


def final_acc(run_dir: str) -> float:
    df = pd.read_csv(os.path.join(run_dir, "trajectory_layers.csv"))
    return float(df.loc[df.epoch == df.epoch.max(), "test_acc"].iloc[0])


def load_arm(results: str, arch: str, arm: str, seed: int) -> pd.DataFrame | None:
    path = os.path.join(results, "round3_decompose", f"a5_{arch}_{arm}_s{seed}_decompose.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def judge_frames(frames: dict) -> tuple[dict, str]:
    """frames: {(arch, seed): dict(none=df, nd=df|None, wo=df|None, cf=df, post=df|None, acc={arm: float})}."""
    out, lines = {}, ["plan 9.56 judgement (tau = 0.95; layers with none-arm data_0.999 < 1 excluded)"]
    p128, p129a, p129b, p130, p131 = {}, {}, [], [], []
    for (arch, seed), f in sorted(frames.items()):
        none, cf = f["none"], f["cf"]
        keep = none.loc[none[f"data_0.999"] >= 1.0, "layer"]
        excl = sorted(set(none.layer) - set(keep))
        base = cf[cf.layer.isin(keep)].set_index("layer")
        line = f"  {arch} s{seed}: {len(keep)} layers kept, excluded {excl}"
        if f.get("nd") is not None:
            nd = f["nd"].set_index("layer").loc[base.index]
            d = (nd[f"out_{TAU}"] - base["nd_pred_95"]).abs().median()
            p128.setdefault(arch, []).append(d < 0.10)
            line += f" | P9.128 median|out_nd - nd_pred| = {d:.3f}"
        if f.get("wo") is not None:
            wo = f["wo"].set_index("layer").loc[base.index]
            d = (wo[f"out_{TAU}"] - base["both_pred_95"]).abs().median()
            p129a.setdefault(arch, []).append(d < 0.10)
            above = (wo[f"out_{TAU}"] > base["ortho_95"]).values
            p129b.extend([(arch, bool(b)) for b in above])
            line += f" | P9.129' median|out_wo - both_pred| = {d:.3f}, out_wo > ortho_none {above.mean():.2f}"
        if f.get("post") is not None:
            r = f["post"]["ratio_95"].values
            p130.extend(list((r >= 0.90) & (r <= 1.10)))
            line += f" | P9.130 ratio in [0.9,1.1]: {np.mean((r >= 0.90) & (r <= 1.10)):.2f} (median {np.median(r):.3f})"
        acc = f.get("acc", {})
        for arm in ("nd", "wo"):
            if arm in acc and "none" in acc:
                ok = acc[arm] >= acc["none"] - 1.0
                p131.append(ok)
                line += f" | acc {arm} {acc[arm]:.2f} vs none {acc['none']:.2f} {'ok' if ok else 'FAIL'}"
        lines.append(line)
    out["P9.128"] = bool(p128) and all(sum(v) >= 2 for v in p128.values()) and len(p128) == 2
    ok129b = all(np.mean([b for a, b in p129b if a == arch]) >= 0.80 for arch in p129a) if p129b else False
    out["P9.129'"] = bool(p129a) and all(sum(v) >= 2 for v in p129a.values()) and len(p129a) == 2 and ok129b
    out["P9.130"] = bool(p130) and float(np.mean(p130)) >= 0.80
    out["P9.131"] = bool(p131) and all(p131)
    out["n_runs"] = {k: len(v) for k, v in p128.items()}
    for k in ("P9.128", "P9.129'", "P9.130", "P9.131"):
        lines.append(f"{k}: {'HOLDS' if out[k] else 'FAILS / incomplete'}")
    return out, "\n".join(lines)


def collect(results: str) -> dict:
    frames = {}
    for arch in ARCHS:
        for seed in SEEDS:
            none = load_arm(results, arch, "none", seed)
            cf_path = os.path.join(results, "t3", f"counterfactual_{arch}_none_s{seed}.csv")
            if none is None or not os.path.exists(cf_path):
                continue
            f = dict(none=none, cf=pd.read_csv(cf_path), acc={})
            for arm in ("nd", "wo"):
                f[arm] = load_arm(results, arch, arm, seed)
            post = os.path.join(results, "t3", f"postrelu_a5_{arch}_wo_s{seed}.csv")
            f["post"] = pd.read_csv(post) if os.path.exists(post) else None
            for arm in ("none", "nd", "wo"):
                run = os.path.join(results, "round3", f"a5_{arch}_{arm}_s{seed}")
                if os.path.exists(os.path.join(run, "trajectory_layers.csv")):
                    f["acc"][arm] = final_acc(run)
            frames[(arch, seed)] = f
    return frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    frames = collect(a.results)
    verdict, text = judge_frames(frames)
    os.makedirs(os.path.join(a.results, "t3"), exist_ok=True)
    with open(os.path.join(a.results, "t3", "judgement.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(a.results, "t3", "judgement.json"), "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
