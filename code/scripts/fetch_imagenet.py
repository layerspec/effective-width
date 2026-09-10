"""Fetch ImageNet-1k (train and/or validation) from Hugging Face into class
folders, pre-resized, for training from scratch (analysis-plan 9.11).

Output layout (torchvision ImageFolder; class index = sorted folder name =
the HF integer label, so train and val agree):

    <out>/train/0000/<name>.JPEG ... <out>/train/0999/...
    <out>/val/0000/...            (50,000 images)

Every image is re-encoded with its SHORTER SIDE resized to --short-side
(default 256, quality 90).  This is a pre-registered implementation choice
(plan 9.11, 2026-09-11, data unseen): it cuts the 150 GB parquet to ~45 GB
of JPEGs and halves decode time, which is the pod's bottleneck; it applies
to every arm identically, and the evaluation transform (Resize(256),
CenterCrop(224)) is unchanged by it.  RandomResizedCrop on a 256-short-side
image is the common "ImageNet-256" variant of the torchvision reference recipe.

--format blob (the default since 2026-09-11) writes one <shard>.blob +
<shard>.idx.npy pair per parquet shard instead of one file per image (see
scripts/imagenet_blob.py: network volumes cannot serve small files fast
enough).  --format folder gives the ImageFolder layout above.

Shards are downloaded one at a time per worker and deleted once written, so
the disk never holds more than ~N_WORKERS parquet shards (~1 GB each) on top
of the JPEGs.  Re-running skips shards whose marker file exists.

    hf auth login            # once; the dataset is gated (instant click-through)
    python scripts/fetch_imagenet.py --out /workspace/imagenet --workers 16
    python scripts/fetch_imagenet.py --out /workspace/imagenet --split val   # val only

The data is under the ImageNet terms of access (non-commercial research).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "ILSVRC/imagenet-1k"
EXPECTED = {"train": 1_281_167, "val": 50_000}


def list_shards(repo: str, split: str) -> list[str]:
    from huggingface_hub import list_repo_files
    prefix = {"train": "data/train-", "val": "data/validation-"}[split]
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {prefix} in {repo}; the naming may have changed")
    return files


def reencode(raw: bytes, short_side: int, quality: int) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    if short_side:
        # JPEG DCT-domain downscale first (fast), then the exact resize
        im.draft("RGB", (short_side * 2, short_side * 2))
        im = im.convert("RGB")
        w, h = im.size
        s = short_side / min(w, h)
        if s < 1:
            im = im.resize((max(short_side, round(w * s)), max(short_side, round(h * s))), Image.BILINEAR)
    else:
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def process_shard(job) -> tuple[str, int]:
    repo, shard, out, split, cache, short_side, quality, fmt = job
    from huggingface_hub import hf_hub_download
    import numpy as np
    import pyarrow.parquet as pq
    marker = os.path.join(out, ".done", split, os.path.basename(shard) + ".done")
    if os.path.exists(marker):
        return shard, -1
    path = hf_hub_download(repo, shard, repo_type="dataset", local_dir=cache)
    n = 0
    pf = pq.ParquetFile(path)
    stem = os.path.splitext(os.path.basename(shard))[0]
    blob_path = os.path.join(out, split, stem + ".blob")
    blob, index = None, []
    if fmt == "blob":
        os.makedirs(os.path.dirname(blob_path), exist_ok=True)
        blob = open(blob_path + ".tmp", "wb")
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        images, labels = batch.column("image").to_pylist(), batch.column("label").to_pylist()
        for rec, label in zip(images, labels):
            raw = rec.get("bytes") if isinstance(rec, dict) else rec
            if raw is None or label is None or label < 0:
                continue
            if fmt == "blob":
                data = reencode(raw, short_side, quality)
                index.append((blob.tell(), len(data), int(label)))
                blob.write(data)
            else:
                name = os.path.basename(rec.get("path") or "") if isinstance(rec, dict) else ""
                name = name or f"{stem}_{n:07d}.JPEG"
                d = os.path.join(out, split, f"{label:04d}")
                os.makedirs(d, exist_ok=True)
                dest = os.path.join(d, os.path.splitext(name)[0] + ".JPEG")
                if not os.path.exists(dest):
                    with open(dest + ".tmp", "wb") as fh:
                        fh.write(reencode(raw, short_side, quality))
                    os.replace(dest + ".tmp", dest)
            n += 1
    if fmt == "blob":
        blob.close()
        np.save(blob_path[: -len(".blob")] + ".idx.npy", np.asarray(index, dtype=np.int64).reshape(-1, 3))
        os.replace(blob_path + ".tmp", blob_path)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as fh:
        fh.write(str(n))
    try:
        os.remove(path)
    except OSError:
        pass
    return shard, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("train", "val", "both"), default="both")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--cache", default=None, help="where shards land before conversion (default <out>/.parquet)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--short-side", type=int, default=256, help="0 keeps the original resolution")
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--limit-shards", type=int, default=None, help="trial run: only this many shards per split")
    p.add_argument("--format", choices=("blob", "folder"), default="blob",
                   help="blob: one <shard>.blob + .idx.npy per shard (scripts/imagenet_blob.py); folder: ImageFolder")
    a = p.parse_args(argv)
    try:
        import huggingface_hub, pyarrow, PIL  # noqa: F401
    except ImportError as e:
        print(f"missing dependency: {e}\n  pip install huggingface_hub pyarrow pillow", file=sys.stderr)
        return 1
    cache = a.cache or os.path.join(a.out, ".parquet")
    os.makedirs(cache, exist_ok=True)
    splits = ["train", "val"] if a.split == "both" else [a.split]
    for split in splits:
        shards = list_shards(a.repo, split)
        if a.limit_shards:
            shards = shards[: a.limit_shards]
        print(f"{split}: {len(shards)} shards -> {a.out}/{split}  (workers {a.workers}, short side {a.short_side})", flush=True)
        jobs = [(a.repo, s, a.out, split, cache, a.short_side, a.quality, a.format) for s in shards]
        t0, total, done = time.time(), 0, 0
        with Pool(a.workers) as pool:
            for shard, n in pool.imap_unordered(process_shard, jobs):
                done += 1
                if n >= 0:
                    total += n
                print(f"  [{done}/{len(shards)}] {os.path.basename(shard)}: {'skipped' if n < 0 else n}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if a.format == "blob":
            from scripts.imagenet_blob import count_images, BlobImageFolder
            count = count_images(os.path.join(a.out, split))
            n_cls = len(BlobImageFolder(os.path.join(a.out, split)).classes)
        else:
            count = sum(len(fs) for _, _, fs in os.walk(os.path.join(a.out, split)))
            n_cls = len([d for d in os.listdir(os.path.join(a.out, split)) if not d.startswith(".")])
        print(f"{split}: {count} images on disk in {n_cls} classes (this run wrote {total})", flush=True)
        if not a.limit_shards and (count != EXPECTED[split] or n_cls != 1000):
            print(f"!! expected {EXPECTED[split]} {split} images in 1000 classes, found {count} in {n_cls}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
