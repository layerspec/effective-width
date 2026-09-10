"""Blob layout of ImageNet (scripts/imagenet_blob.py, plan 9.11)."""
import io
import os
import pickle

import numpy as np
from PIL import Image

from scripts.imagenet_blob import BlobImageFolder, count_images, imagenet_split, is_blob_dir


def _write_blob(d, stem, labels, seed=0):
    rng = np.random.default_rng(seed)
    idx, off = [], 0
    with open(os.path.join(d, stem + ".blob"), "wb") as fh:
        for lab in labels:
            b = io.BytesIO()
            Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(b, format="JPEG")
            data = b.getvalue()
            fh.write(data); idx.append((off, len(data), lab)); off += len(data)
    np.save(os.path.join(d, stem + ".idx.npy"), np.asarray(idx, dtype=np.int64))


def test_blob_folder_reads_like_imagefolder(tmp_path):
    d = tmp_path / "train"; d.mkdir()
    _write_blob(d, "train-00000", [5, 0, 5, 7])
    _write_blob(d, "train-00001", [7, 0], seed=1)
    assert is_blob_dir(str(d)) and count_images(str(d)) == 6
    ds = BlobImageFolder(str(d))
    assert ds.classes == ["0000", "0005", "0007"] and ds.class_to_idx["0007"] == 2
    assert ds.targets == [1, 0, 1, 2, 2, 0] and len(ds) == 6
    img, y = ds[3]
    assert img.size == (50, 40) and img.mode == "RGB" and y == 2
    img5, _ = ds[5]                                      # second blob file
    assert img5.size == (50, 40)
    ds2 = pickle.loads(pickle.dumps(ds))                 # worker processes get a fresh handle table
    assert ds2._fh == {} and ds2[0][1] == 1
    assert isinstance(imagenet_split(str(tmp_path), "train"), BlobImageFolder)
