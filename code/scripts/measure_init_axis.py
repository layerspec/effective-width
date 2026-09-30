"""A33 (analysis-plan 9.23): the initialisation as an axis of the measurement.  Every family gets the random networks
of each initialisation its public weights were trained from; only the missing ones are measured here.

  gamma0      ResNet-50 with the last batch-norm weight of every block at zero (timm's default; the public random
              networks of ResNet-50 are torchvision's, gamma = 1)      protocol of layerspec.run, 16 positions
                  -> <out>/gamma0/seed<k>/timm_resnet50.tv_in1k_random_layers.csv
  layerscale  DeiT-III, small and base, LayerScale at 1e-4 (the value of the DeiT-III code; timm's constructor, which
              the A30 random networks of deit3_small used, has 1e-6)   protocol of A30, 32 tokens
                  -> <out>/layerscale/<model>_random_s<k>_layers.csv
  jax         ViT-S/16 and ViT-B/16 with the initialisation of the JAX code the augreg / orig / sam weights were
              trained with (Xavier-uniform weights, MLP biases N(0, 1e-6))   protocol of A30, 32 tokens
                  -> <out>/jax/<model>_random_s<k>_layers.csv

Wrapper: layerspec and the two measurement scripts it borrows from are not modified.  Seeds 1-3, ImageNet input
statistics (random networks), CPU.  Resumable: a finished file is skipped.

    python scripts/measure_init_axis.py --out ../results/init_axis
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

SEEDS = (1, 2, 3)
LAYERSCALE_INIT = 1e-4
JOBS = (   # kind, timm architecture, constructor arguments, model spec handed to layerspec.run, transformer protocol?
    ("jax", "vit_small_patch16_224", dict(weight_init="jax"), "timm:vit_small_patch16_224_random", True),
    ("layerscale", "deit3_small_patch16_224", dict(init_values=LAYERSCALE_INIT), "timm:deit3_small_patch16_224_random", True),
    ("gamma0", "resnet50", dict(zero_init_last=True), "timm:resnet50.tv_in1k_random", False),
    ("jax", "vit_base_patch16_224", dict(weight_init="jax"), "timm:vit_base_patch16_224_random", True),
    ("layerscale", "deit3_base_patch16_224", dict(init_values=LAYERSCALE_INIT), "timm:deit3_base_patch16_224_random", True),
)


def build_model(arch: str, kwargs: dict, seed: int):
    """The random network of one job: the seed is set immediately before the constructor draws the weights."""
    import timm
    import torch
    torch.manual_seed(seed)
    return timm.create_model(arch, pretrained=False, **kwargs)


def jobs(out: str) -> list[dict]:
    j = []
    for kind, arch, kwargs, spec, transformer in JOBS:
        for s in SEEDS:
            path = (os.path.join(out, kind, f"{arch}_random_s{s}") if transformer
                    else os.path.join(out, kind, f"seed{s}", "timm_resnet50.tv_in1k_random"))
            j.append(dict(kind=kind, arch=arch, kwargs=kwargs, spec=spec, transformer=transformer, seed=s, path=path))
    return j


def measure(job: dict, a) -> dict:
    import layerspec.run as run
    tag = ", ".join(f"{k}={v}" for k, v in job["kwargs"].items())

    def build(name, pretrained=True, seed=0):
        return build_model(job["arch"], job["kwargs"], seed), f"random_init(seed={seed}, {tag})"
    saved = run._models.build
    run._models.build = build
    try:
        args = SimpleNamespace(data=a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit,
                               seed=job["seed"], preprocess="crop", device=a.device,
                               positions=32 if job["transformer"] else 16, conv_only=job["transformer"],
                               bn=False, max_batches=None, min_n_over_C=50.0)
        t0 = time.time()
        layers, conv, spectra, meta = run.measure_one(job["spec"], args)
    finally:
        run._models.build = saved
    os.makedirs(os.path.dirname(job["path"]), exist_ok=True)
    np.savez_compressed(job["path"] + "_spectra.npz", **spectra)
    if not job["transformer"]:
        conv.to_csv(job["path"] + "_convergence.csv", index=False)
    layers.to_csv(job["path"] + "_layers.csv.tmp", index=False)
    os.replace(job["path"] + "_layers.csv.tmp", job["path"] + "_layers.csv")      # the file that marks the job done
    meta.update({"kind": job["kind"], "arch": job["arch"], "init": job["kwargs"], "seconds": round(time.time() - t0, 1)})
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/init_axis")
    p.add_argument("--only", choices=("gamma0", "layerscale", "jax"), default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for job in jobs(a.out):
        if a.only and job["kind"] != a.only:
            continue
        name = os.path.relpath(job["path"], a.out)
        if os.path.exists(job["path"] + "_layers.csv"):
            print(f"=== {name} done, skipping ===", flush=True); continue
        print(f"=== {name} start {time.strftime('%F %T')} ===", flush=True)
        manifest[name] = measure(job, a)
        with open(manifest_path + ".tmp", "w") as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(manifest_path + ".tmp", manifest_path)
        print(f"=== {name} end, {manifest[name]['seconds']:.0f} s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
