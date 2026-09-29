"""B+ controls on the measurement itself (analysis-plan 9.20), with layerspec.run's protocol and nothing else changed:

  renorm   the five public weights that were trained with input statistics other than ImageNet's and were measured
           with ImageNet's (MobileNetV3 x3, EfficientNet-B0 x2), re-measured with their own (timm pretrained_cfg)
               -> <out>/renorm/timm_<weight>_layers.csv
  gamma1   random networks of the five ResNet families whose timm constructor zero-initialises the last BN of every
           block, rebuilt with zero_init_last=False (gamma = 1, the torchvision default), seeds 1-3
               -> <out>/gamma1/seed<k>/timm_<family>_random_layers.csv

Options are those of the original measurements (6,400 images, 16 positions, seed 0 for weights, activations hooked);
the device is the CPU.  Resumable: a finished file is skipped.

    python scripts/measure_controls.py --out ../results/bplus_controls
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

RENORM = ["mobilenetv3_large_100.ra4_e3600_r224_in1k", "mobilenetv3_large_100.miil_in21k_ft_in1k",
          "tf_mobilenetv3_large_100.in1k", "efficientnet_b0.ra4_e3600_r224_in1k", "tf_efficientnet_b0.ap_in1k"]
GAMMA1 = ["resnet18.tv_in1k", "resnet34.tv_in1k", "resnet101.tv_in1k", "wide_resnet50_2.tv_in1k", "resnext50_32x4d.tv_in1k"]
SEEDS = (1, 2, 3)
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def jobs(out: str) -> list[dict]:
    j = [dict(kind="renorm", base=w, seed=0, path=os.path.join(out, "renorm", f"timm_{w}")) for w in RENORM]
    j += [dict(kind="gamma1", base=f, seed=s, path=os.path.join(out, "gamma1", f"seed{s}", f"timm_{f}_random"))
          for f in GAMMA1 for s in SEEDS]                       # small families first within each seed order below
    return j


def measure(job: dict, a) -> dict:
    import timm
    import torch
    import layerspec.data as _data
    import layerspec.run as run
    if job["kind"] == "renorm":
        cfg = timm.get_pretrained_cfg(job["base"])
        norm, spec = (tuple(cfg.mean), tuple(cfg.std)), f"timm:{job['base']}"
        build = None
    else:
        norm, spec = IMAGENET, f"timm:{job['base']}_random"

        def build(name, pretrained=True, seed=0):
            torch.manual_seed(seed)
            return timm.create_model(job["base"], pretrained=False, zero_init_last=False), f"random_init(seed={seed}, zero_init_last=False)"
    saved = (_data.IMAGENET_MEAN, _data.IMAGENET_STD, run._models.build)
    _data.IMAGENET_MEAN, _data.IMAGENET_STD = norm
    if build is not None:
        run._models.build = build
    try:
        args = SimpleNamespace(data=a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit,
                               seed=job["seed"], preprocess="crop", device=a.device, positions=a.positions,
                               conv_only=False, bn=False, max_batches=None, min_n_over_C=50.0)
        t0 = time.time()
        layers, conv, spectra, meta = run.measure_one(spec, args)
    finally:
        _data.IMAGENET_MEAN, _data.IMAGENET_STD, run._models.build = saved
    os.makedirs(os.path.dirname(job["path"]), exist_ok=True)
    np.savez_compressed(job["path"] + "_spectra.npz", **spectra)
    conv.to_csv(job["path"] + "_convergence.csv", index=False)
    layers.to_csv(job["path"] + "_layers.csv.tmp", index=False)
    os.replace(job["path"] + "_layers.csv.tmp", job["path"] + "_layers.csv")      # the file that marks the job done
    meta.update({"kind": job["kind"], "mean": list(norm[0]), "std": list(norm[1]), "seconds": round(time.time() - t0, 1)})
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/bplus_controls")
    p.add_argument("--only", choices=("renorm", "gamma1"), default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=16)
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
