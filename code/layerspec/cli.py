"""Command line: layerspec profile|decompose --model NAME --data DIR --out FILE.csv"""
from __future__ import annotations

import argparse
import sys

from . import api


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help='torchvision name, "timm:<name>", or "<name>_random"')
    p.add_argument("--data", default="synthetic", help="image directory (any layout) or 'synthetic'")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit", type=int, default=None, help="use at most this many images")
    p.add_argument("--seed", type=int, default=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="layerspec", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("profile", help="k*(tau)/r_max for every convolution output")
    _common(pp)
    pp.add_argument("--positions", type=int, default=16, help="sampled spatial positions per image")
    pp.add_argument("--min-n-over-C", type=float, default=50.0)
    pd_ = sub.add_parser("decompose", help="out / kernel / data / ortho per dense convolution")
    _common(pd_)
    pd_.add_argument("--positions", type=int, default=48)
    pd_.add_argument("--rank-tol", type=float, default=1e-6)
    a = p.parse_args(argv)

    model = api.load_model(a.model, seed=a.seed)
    loader = api.image_loader(None if a.data == "synthetic" else a.data, batch_size=a.batch_size,
                              workers=a.workers, image_size=a.image_size, limit=a.limit, seed=a.seed)
    if a.cmd == "profile":
        res = api.profile(model, loader, device=a.device, positions=a.positions,
                          min_n_over_C=a.min_n_over_C, seed=a.seed, progress=True)
        s = res.summary()
        print(f"{a.model}: {s['L']} dense conv layers pass the gate; median k*(0.95)/r_max {s['median_level']:.3f}; "
              f"rho(depth) {s['rho_depth']:+.2f}; max k*({s['rmax_check_tau']:g})/r_max {s['rmax_check_max']:.3f}"
              + ("" if s["rmax_check_ok"] else "  ** exceeds 1: r_max is wrong for some layer **"))
        for line in res.warnings():
            print(line)
    else:
        res = api.decompose(model, loader, device=a.device, positions=a.positions,
                            rank_tol=a.rank_tol, seed=a.seed)
        c = res.check()
        print(f"{a.model}: {len(res.table)} dense conv layers; Ostrowski {c['ostrowski'][0]}/{c['ostrowski'][1]}; "
              f"kappa median {c['kappa_median']:.1f}")
        if c["n_below_50"] > 0:
            print(f"!! {c['n_below_50']}/{len(res.table)} dense convolutions have n/d < 50; "
                  f"their four quantities are sampling artefacts")
    if a.cmd == "profile" and (res.table.kind == "linear").any():
        s2 = res.summary(kinds=("linear",))
        print(f"{a.model}: {s2['L']} linear layers pass the gate; median k*(0.95)/r_max {s2['median_level']:.3f}; "
              f"rho(depth) {s2['rho_depth']:+.2f}")
    res.to_csv(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
