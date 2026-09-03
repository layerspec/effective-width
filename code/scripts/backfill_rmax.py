"""Add the attainable-rank column to result CSVs written before it was recorded.

A conv layer's output at one spatial position is a linear map of its
receptive-field patch, so the channel covariance has rank at most

    r_max = groups * min((C_in/groups) * kh * kw, C_out/groups)

(the weight matrix is block-diagonal over groups and the blocks have disjoint
supports, so their ranks add).  Normalising k* by C_out alone charges a layer
for width it could not have used.  ResNet-50 is the case that matters: its 1x1
expansions have C_in = C_out/4, so k*/C cannot exceed 0.25 however well the
network is trained, and the "wide layers use a tenth of their width" reading of
the round-one numbers was an artefact of that denominator.

`layerspec.hooks` records r_max at run time now.  This script exists only to
retrofit runs made before it did.  Two modes, so the machine holding the
results does not need torch:

    python backfill_rmax.py dump  --archs resnet50 vgg16_bn --out rmax.json
    python backfill_rmax.py apply --table rmax.json --results results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def dump(archs, out_path, image_size=224):
    import torch
    import torch.nn as nn
    import torchvision

    table = {}
    for arch in archs:
        model = getattr(torchvision.models, arch)(weights=None).eval()
        per_arch, calls = {}, {}
        model.register_forward_pre_hook(lambda _m, _i: calls.clear())
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue

            def hook(_m, _inp, _out, _name=name):
                i = calls.get(_name, 0)
                calls[_name] = i + 1
                kh, kw = _m.kernel_size
                g = _m.groups
                per = min((_m.in_channels // g) * kh * kw, _m.out_channels // g)
                per_arch[f"{_name}#{i}"] = {
                    "in_channels": _m.in_channels,
                    "kernel": [kh, kw],
                    "groups": g,
                    "out_channels": _m.out_channels,
                    "r_max": g * per,
                }

            mod.register_forward_hook(hook)
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size))
        table[arch] = per_arch
        print(f"  {arch}: {len(per_arch)} conv sites")

    with open(out_path, "w") as f:
        json.dump(table, f, indent=1)
    print(f"wrote {out_path}")
    return 0


TAUS = ("0.9", "0.95", "0.99", "0.999")


def apply(table_path, results_dir):
    import pandas as pd

    table = json.load(open(table_path))
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(("_layers.csv", "_convergence.csv")):
            continue
        model = fn.rsplit("_", 1)[0].replace("_layers", "").replace("_convergence", "")
        arch = model[: -len("_random")] if model.endswith("_random") else model
        if arch not in table:
            print(f"  skip {fn}: no entry for architecture {arch!r}")
            continue
        path = os.path.join(results_dir, fn)
        df = pd.read_csv(path)

        def lookup(r):
            if r["kind"] != "conv":
                return None
            e = table[arch].get(f"{r['layer']}#{int(r['call_index'])}")
            return e["r_max"] if e else None

        df["r_max"] = df.apply(lookup, axis=1)
        missing = int(df[df["kind"] == "conv"]["r_max"].isna().sum())
        for tau in TAUS:
            col = f"k_star_{tau}"
            if col in df.columns:
                df[f"k_star_rmax_{tau}"] = df[col] / df["r_max"]
        df["r_max_over_C"] = df["r_max"] / df["C"]
        df.to_csv(path, index=False)
        print(f"  {fn}: filled {int(df['r_max'].notna().sum())} rows"
              + (f", {missing} conv rows UNMATCHED" if missing else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump", help="compute r_max per layer (needs torch)")
    d.add_argument("--archs", nargs="+", required=True)
    d.add_argument("--out", default="rmax.json")
    a = sub.add_parser("apply", help="merge into result CSVs (needs pandas only)")
    a.add_argument("--table", default="rmax.json")
    a.add_argument("--results", default="results")
    args = p.parse_args(argv)
    return dump(args.archs, args.out) if args.mode == "dump" \
        else apply(args.table, args.results)


if __name__ == "__main__":
    raise SystemExit(main())
