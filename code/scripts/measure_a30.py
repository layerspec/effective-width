"""A30 (analysis-plan 9.19): the ruler on two transformer families, ViT-S/16 and ViT-B/16, public weights against
default initialisations.  layerspec.run's measurement, with one change: the input is normalised with the mean and
standard deviation each weight was trained with (timm's pretrained_cfg; augreg / orig / sam weights use 0.5, not the
ImageNet statistics).  Random networks are measured with the ImageNet statistics.  The geometry of the protocol
(Resize 256 + CenterCrop 224) and the images (the section-6 subset) are unchanged; 32 tokens per image.

    python scripts/measure_a30.py --out ../results/a30                       # everything, skipping what is done
    python scripts/measure_a30.py --out ../results/a30 --models timm:deit_small_patch16_224.fb_in1k

Writes <out>/<name>_layers.csv (the schema of layerspec.run), <name>_spectra.npz and manifest rows in <out>/a30.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPERVISED = {
    "vit_s": ["vit_small_patch16_224.augreg_in1k", "vit_small_patch16_224.augreg_in21k_ft_in1k",
              "deit_small_patch16_224.fb_in1k", "deit3_small_patch16_224.fb_in1k",
              "deit3_small_patch16_224.fb_in22k_ft_in1k"],
    "vit_b": ["vit_base_patch16_224.augreg_in1k", "vit_base_patch16_224.augreg_in21k_ft_in1k",
              "vit_base_patch16_224.orig_in21k_ft_in1k", "vit_base_patch16_224.sam_in1k",
              "deit_base_patch16_224.fb_in1k", "deit3_base_patch16_224.fb_in1k"],
}
SELF_SUPERVISED = {"vit_s": ["vit_small_patch16_224.dino"], "vit_b": ["vit_base_patch16_224.dino", "vit_base_patch16_224.mae"]}
RANDOM = {"vit_s": "vit_small_patch16_224", "vit_b": "vit_base_patch16_224", "deit3_s": "deit3_small_patch16_224"}
RANDOM_SEEDS = (1, 2, 3)
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def jobs() -> list[tuple[str, int, str]]:
    """(model spec, seed, output name); small family first, random networks after the weights of their family."""
    out = []
    for fam in ("vit_s", "vit_b"):
        out += [(f"timm:{m}", 0, m) for m in SUPERVISED[fam] + SELF_SUPERVISED[fam]]
        out += [(f"timm:{RANDOM[fam]}_random", s, f"{RANDOM[fam]}_random_s{s}") for s in RANDOM_SEEDS]
        if fam == "vit_s":
            out += [(f"timm:{RANDOM['deit3_s']}_random", s, f"{RANDOM['deit3_s']}_random_s{s}") for s in RANDOM_SEEDS]
    return out


def normalisation(spec: str):
    """The (mean, std) a weight was trained with; ImageNet's for a random network."""
    base = spec.split(":", 1)[1]
    if base.endswith("_random"):
        return IMAGENET
    import timm
    cfg = timm.get_pretrained_cfg(base)
    return tuple(cfg.mean), tuple(cfg.std)


def measure(spec: str, seed: int, name: str, a) -> dict:
    import layerspec.data as _data
    from layerspec.run import measure_one
    mean, std = a.force_norm or normalisation(spec)
    saved = (_data.IMAGENET_MEAN, _data.IMAGENET_STD)
    _data.IMAGENET_MEAN, _data.IMAGENET_STD = mean, std
    try:
        args = SimpleNamespace(data=a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit,
                               seed=seed, preprocess="crop", device=a.device, positions=a.positions, conv_only=True,
                               bn=False, max_batches=None, min_n_over_C=50.0)
        t0 = time.time()
        layers, _conv, spectra, meta = measure_one(spec, args)
    finally:
        _data.IMAGENET_MEAN, _data.IMAGENET_STD = saved
    layers.to_csv(os.path.join(a.out, f"{name}_layers.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, f"{name}_spectra.npz"), **spectra)
    ok = layers[layers.n_over_C_ok.astype(bool) & (layers.kind == "linear")]
    meta.update({"name": name, "mean": list(mean), "std": list(std), "linear_layers_passing": int(len(ok)),
                 "level_k95_rmax": float(ok["k_star_rmax_0.95"].median()) if len(ok) else None,
                 "seconds": round(time.time() - t0, 1)})
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/a30")
    p.add_argument("--models", nargs="+", default=None, help="model specs (timm:<name>[_random]); default: the registered list")
    p.add_argument("--seed", type=int, default=0, help="seed of a model given with --models")
    p.add_argument("--name", default=None, help="output name of a single model given with --models")
    p.add_argument("--imagenet-norm", dest="force_norm", action="store_const", const=IMAGENET, default=None,
                   help="P9.55: measure with the ImageNet statistics whatever the weight was trained with")
    p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    todo = jobs() if a.models is None else [(m, a.seed, a.name or m.split(":", 1)[1]) for m in a.models]
    manifest_path = os.path.join(a.out, "a30.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for spec, seed, name in todo:
        if os.path.exists(os.path.join(a.out, f"{name}_layers.csv")):
            print(f"=== {name} done, skipping ===", flush=True); continue
        print(f"=== {name} start {time.strftime('%F %T')} ===", flush=True)
        meta = measure(spec, seed, name, a)
        manifest[name] = meta
        with open(manifest_path + ".tmp", "w") as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(manifest_path + ".tmp", manifest_path)
        print(f"=== {name} end: {meta['linear_layers_passing']} linear layers pass, {meta['seconds']:.0f} s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
