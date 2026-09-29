"""A28 (analysis-plan 9.17): a copy of the ImageNet blob dataset with every image shrunk to a short side of 80
pixels, for the 64-pixel cells.  Shards, image order and labels are kept, so the indices of scripts/a28_lists.py
address the same images in both copies.

    <out>/<split>/<shard>.blob, <shard>.idx.npy     as scripts/imagenet_blob.py reads them

Antialiased (PIL reduces the JPEG at decode time, then LANCZOS), re-encoded at quality 95.  One process per shard.

    python scripts/a28_downsample.py --data /workspace/imagenet --out /workspace/imagenet80 --workers 32
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

SHORT = 80


def shrink(jpeg: bytes, short: int = SHORT) -> bytes:
    img = Image.open(io.BytesIO(jpeg))
    img.draft("RGB", (2 * short, 2 * short))             # DCT-domain reduction, never below twice the target
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    if s < 1:
        img = img.resize((max(short, round(w * s)), max(short, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def convert(job) -> int:
    idx_file, out_dir, short = job
    name = os.path.basename(idx_file)[: -len(".idx.npy")]
    dst_idx = os.path.join(out_dir, name + ".idx.npy")
    if os.path.exists(dst_idx):
        return 0
    table = np.load(idx_file)
    new = np.zeros_like(table)
    with open(idx_file[: -len(".idx.npy")] + ".blob", "rb") as src, open(os.path.join(out_dir, name + ".blob.tmp"), "wb") as dst:
        for i, (off, length, label) in enumerate(table):
            src.seek(int(off))
            data = shrink(src.read(int(length)), short)
            new[i] = (dst.tell(), len(data), label)
            dst.write(data)
    os.replace(os.path.join(out_dir, name + ".blob.tmp"), os.path.join(out_dir, name + ".blob"))
    np.save(dst_idx + ".tmp.npy", new)
    os.replace(dst_idx + ".tmp.npy", dst_idx)             # the index appears last: a shard with an index is complete
    return len(table)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--short", type=int, default=SHORT)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args(argv)
    for split in a.splits:
        files = sorted(glob.glob(os.path.join(a.data, split, "*.idx.npy")))
        if not files:
            raise SystemExit(f"no blob shards in {os.path.join(a.data, split)}")
        out_dir = os.path.join(a.out, split)
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(f, out_dir, a.short) for f in files]
        if a.workers > 1:
            with Pool(a.workers) as pool:
                n = sum(pool.imap_unordered(convert, jobs))
        else:
            n = sum(convert(j) for j in jobs)
        print(f"{split}: {len(files)} shards, {n} images written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
