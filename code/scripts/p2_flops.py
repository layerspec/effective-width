#!/usr/bin/env python
"""Training and inference compute of the controller runs vs A18's arms (2026-09-20, for the hostile review).

VGG-16 BN on 32x32 CIFAR: conv i has 9 * C_in * C_out * H_i * W_i multiply-adds per image, resolution 32/16/8/4/2 in the
five stages (pool after conv 1, 3, 6, 9, 12); the 1-layer classifier is negligible. Training cost per image is taken as
3x the forward MACs (forward + 2x backward), the same convention as Yuan et al. 2023. A run's training cost is the sum
over its 100 epochs of 50,000 images x the MACs of the widths in force in that epoch (from realloc.json); the two
calibration passes (2 x 6,400 images forward) are added explicitly. Arms: full = 100 epochs at full width; ruler /
ruler_act = 100 epochs full (the measurement pass) + 100 epochs at the ruler widths; uniform = 100 epochs at 0.339.

    python scripts/p2_flops.py --results ../results
"""
import argparse
import glob
import json
import os

FULL = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
RES = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
UNIFORM = [37, 37, 75, 75, 149, 149, 149, 298, 298, 298, 298, 298, 298]
N_TRAIN, EPOCHS, CALIB = 50_000, 100, 6_400


def macs(widths):
    cin, total = 3, 0
    for w, r in zip(widths, RES):
        total += 9 * cin * w * r * r
        cin = w
    return total + widths[-1] * 10


def train_cost(schedule):
    """schedule: list of (n_epochs, widths)."""
    return sum(3 * macs(w) * N_TRAIN * n for n, w in schedule)


def controller_cost(run_dir):
    acts = json.load(open(os.path.join(run_dir, "realloc.json")))
    sched, prev_epoch, widths = [], 0, acts[0]["before"]
    for a in acts:
        sched.append((a["epoch"] - prev_epoch, widths))
        prev_epoch, widths = a["epoch"], a["after"]
    sched.append((EPOCHS - prev_epoch, widths))
    calib = sum(macs(a["before"]) * CALIB for a in acts)       # forward-only statistics passes
    return train_cost(sched) + calib, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    W = json.load(open(os.path.join(a.results, "a18", "widths.json")))["widths"]["ruler"]
    Wa = json.load(open(os.path.join(a.results, "a18", "widths_act.json")))["widths"]["ruler"]
    full = train_cost([(EPOCHS, FULL)])
    arms = {
        "full (1 pass)": (full, macs(FULL)),
        "ruler_act (measure full + retrain)": (full + train_cost([(EPOCHS, Wa)]), macs(Wa)),
        "ruler (measure full + retrain)": (full + train_cost([(EPOCHS, W)]), macs(W)),
        "uniform 0.339 (1 pass)": (train_cost([(EPOCHS, UNIFORM)]), macs(UNIFORM)),
    }
    print(f"{'arm / run':44s} {'train MACs':>12s} {'vs full':>8s} {'infer MACs/img':>15s} {'vs full':>8s}")
    for k, (t, m) in arms.items():
        print(f"{k:44s} {t / 1e15:9.2f} PMAC {t / full:8.2f} {m / 1e6:12.0f} M {m / macs(FULL):8.2f}")
    print()
    groups = {}
    for d in sorted(glob.glob(os.path.join(a.results, "p2", "*_s[0-9]"))):
        if not os.path.exists(os.path.join(d, "realloc.json")):
            continue
        name = os.path.basename(d)
        t, sched = controller_cost(d)
        m = macs(sched[-1][1])
        groups.setdefault(name.rsplit("_s", 1)[0], []).append((name, t, m, sched))
    for g, runs in groups.items():
        for name, t, m, sched in runs:
            print(f"{name:44s} {t / 1e15:9.2f} PMAC {t / full:8.2f} {m / 1e6:12.0f} M {m / macs(FULL):8.2f}")
        ts = [t for _, t, _, _ in runs]; ms = [m for _, _, m, _ in runs]
        print(f"  -> {g}: train {sum(ts) / len(ts) / full:.2f} of full, inference {sum(ms) / len(ms) / macs(FULL):.2f} of full "
              f"(epochs at full width before the first action: {runs[0][3][0][0]})\n")


if __name__ == "__main__":
    main()
