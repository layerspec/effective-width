"""Plan 9.35 (P9.93-P9.95): the sign of the addition step at gamma = 1 initialisation.

Measures x (stream in), f (branch), y = x + f and z = ReLU(y) for every residual block of a random
ResNet-34 and ResNeXt-50 (timm architecture, zero_init_last=False, seed 1), reusing measure_residual.measure,
then judges the registered predictions.

Usage: cd code && python scripts/residual_sign_rule.py [--judge] [--out ../results/residual_sign]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_residual import measure  # noqa: E402

MODELS = [("resnet34.tv_in1k", 1), ("resnext50_32x4d.tv_in1k", 1)]


def judge(out: str) -> dict:
    V = {}
    frames = {}
    for base, seed in MODELS:
        p = os.path.join(out, f"timm_{base}_random_s{seed}_blocks.csv")
        if os.path.exists(p):
            frames[base] = pd.read_csv(p)
    for base, d in frames.items():
        for stage in ["layer1", "layer2", "layer3", "layer4"]:
            g = d[d.block.str.startswith(stage + ".")].sort_values("block_index")
            print(f"  {base:26s} {stage}: x={g.k95_x.tolist()} f={g.k95_f.tolist()} y={g.k95_y.tolist()} z={g.k95_z.tolist()} ok={int(g.ok.sum())}/{len(g)}")
    if "resnet34.tv_in1k" in frames:
        d = frames["resnet34.tv_in1k"]; l3 = d[d.block.str.startswith("layer3.")]; l1 = d[d.block.str.startswith("layer1.")]
        V["P9.93"] = bool((l3.k95_f < l3.k95_x).sum() >= 4 and (l3.k95_z < l3.k95_x).sum() >= 4 and (l1.k95_f > l1.k95_x).sum() >= 2)
        print(f"  P9.93 (ResNet-34 layer3: f<x {(l3.k95_f < l3.k95_x).sum()}/6, z<x {(l3.k95_z < l3.k95_x).sum()}/6; layer1 f>x {(l1.k95_f > l1.k95_x).sum()}/3) -> {'holds' if V['P9.93'] else 'fails'}")
    if "resnext50_32x4d.tv_in1k" in frames:
        d = frames["resnext50_32x4d.tv_in1k"]; l3 = d[d.block.str.startswith("layer3.")]
        V["P9.94"] = bool((l3.k95_f >= l3.k95_x).sum() >= 4 and (l3.k95_z >= l3.k95_x).sum() >= 4)
        print(f"  P9.94 (ResNeXt-50 layer3: f>=x {(l3.k95_f >= l3.k95_x).sum()}/6, z>=x {(l3.k95_z >= l3.k95_x).sum()}/6) -> {'holds' if V['P9.94'] else 'fails'}")
    if frames:
        allb = pd.concat(frames.values()); allb = allb[allb.ok]
        sz = np.sign(allb.k95_z - allb.k95_x); sf = np.sign(allb.k95_f - allb.k95_x)
        agree = ((sz == sf) | (sz == 0) | (sf == 0)).mean()
        relu = ((allb.k95_z - allb.k95_y).abs() <= 0.1 * allb.k95_y).mean()
        V["P9.95"] = bool(agree >= 0.8 and relu >= 0.8)
        print(f"  P9.95 (sign agreement {agree:.0%} over {len(allb)} ok blocks; ReLU step within 10% in {relu:.0%}) -> {'holds' if V['P9.95'] else 'fails'}")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/imagenet_val_6400"); p.add_argument("--out", default="../results/residual_sign")
    p.add_argument("--judge", action="store_true"); p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=32); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    if a.judge:
        judge(a.out); return 0
    os.makedirs(a.out, exist_ok=True)
    for base, seed in MODELS:
        name = f"timm_{base}_random_s{seed}"; path = os.path.join(a.out, f"{name}_blocks.csv")
        if os.path.exists(path):
            print(f"=== {name} done, skipping ===", flush=True); continue
        print(f"=== {name} start {time.strftime('%F %T')} ===", flush=True); t0 = time.time()
        rows, spectra = measure(base, seed, a)
        np.savez_compressed(os.path.join(a.out, f"{name}_spectra.npz"), **spectra)
        rows.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
        print(f"=== {name} end: {len(rows)} blocks, {time.time() - t0:.0f} s ===", flush=True)
    judge(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
