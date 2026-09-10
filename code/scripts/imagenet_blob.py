"""ImageNet as a few large files instead of 1.28 M small ones (plan 9.11).

RunPod network volumes (MooseFS) serve ~1,400 small-file opens per second
across the pod, cold or warm -- a third of what one ResNet-18 training run
consumes -- but 5,600+ random reads per second from one large file (measured
2026-09-11 on the A23 pod).  So `fetch_imagenet.py --format blob` writes, per
parquet shard,

    <out>/<split>/<shard>.blob      the re-encoded JPEG bytes, concatenated
    <out>/<split>/<shard>.idx.npy   int64 array [n, 3]: offset, length, label

and `BlobImageFolder` reads them with one seek + read per image.  It exposes
the same `classes` / `class_to_idx` / `targets` / `__getitem__` surface that
`torchvision.datasets.ImageFolder` does, so the training and measurement code
is indifferent to the format.  Label integers are the HF labels (0..999),
identical between train and val by construction.
"""
from __future__ import annotations

import glob
import io
import os

import numpy as np
from PIL import Image


def is_blob_dir(root: str) -> bool:
    return bool(glob.glob(os.path.join(root, "*.idx.npy")))


def count_images(root: str) -> int:
    return int(sum(np.load(f, mmap_mode="r").shape[0] for f in glob.glob(os.path.join(root, "*.idx.npy"))))


class BlobImageFolder:
    def __init__(self, root: str, transform=None):
        self.root, self.transform = root, transform
        idx_files = sorted(glob.glob(os.path.join(root, "*.idx.npy")))
        if not idx_files:
            raise FileNotFoundError(f"no *.idx.npy in {root}")
        self.blobs = [f[: -len(".idx.npy")] + ".blob" for f in idx_files]
        parts = [np.load(f) for f in idx_files]
        self.file_id = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(parts)])
        table = np.concatenate(parts)
        self.offsets, self.lengths = table[:, 0], table[:, 1]
        labels = table[:, 2]
        uniq = np.unique(labels)
        self.classes = [f"{int(c):04d}" for c in uniq]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        remap = np.full(int(uniq.max()) + 1, -1, dtype=np.int64)
        remap[uniq] = np.arange(len(uniq))
        self.targets = remap[labels].tolist()
        self._fh: dict[int, io.BufferedReader] = {}

    def __len__(self):
        return len(self.targets)

    def _handle(self, i: int):
        fh = self._fh.get(i)
        if fh is None:                       # opened lazily, once per worker process
            fh = self._fh[i] = open(self.blobs[i], "rb")
        return fh

    def __getitem__(self, i: int):
        fh = self._handle(int(self.file_id[i]))
        fh.seek(int(self.offsets[i]))
        img = Image.open(io.BytesIO(fh.read(int(self.lengths[i])))).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]

    def __getstate__(self):                  # file handles do not cross fork/spawn
        d = self.__dict__.copy()
        d["_fh"] = {}
        return d


def imagenet_split(root: str, split: str, transform=None):
    """ImageFolder or BlobImageFolder for <root>/<split>, whichever is on disk."""
    d = os.path.join(root, split)
    if is_blob_dir(d):
        return BlobImageFolder(d, transform)
    from torchvision import datasets
    return datasets.ImageFolder(d, transform=transform)
