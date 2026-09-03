"""Data sources.

Real runs use the ImageNet-1k validation split (50,000 images).  The pipeline
is deliberately agnostic about how it got there: point ``--data`` at any
directory laid out as ``root/<class>/<image>``.

``synthetic`` exists so the pipeline can be exercised end-to-end with no
download at all -- useful for smoke tests and for checking that a rented
machine is set up correctly before paying for the real pass.  Its numbers are
meaningless as science and are tagged as such in the output.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticImages(Dataset):
    """Pink-ish noise with 1/f amplitude falloff, roughly natural-image second-order
    statistics.  Enough to exercise the pipeline; not a substitute for real data."""

    def __init__(self, n: int = 512, size: int = 224, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        s = self.size
        white = torch.randn(3, s, s, generator=g)
        spec = torch.fft.rfft2(white)
        fy = torch.fft.fftfreq(s).unsqueeze(1)
        fx = torch.fft.rfftfreq(s).unsqueeze(0)
        radius = torch.sqrt(fy**2 + fx**2)
        radius[0, 0] = 1.0
        img = torch.fft.irfft2(spec / radius, s=(s, s))
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img, 0


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp", ".webp", ".ppm", ".tif", ".tiff"}


class FlatImageFolder(Dataset):
    """Every image under `root`, recursively, in sorted order. Labels are not read.

    Sorted order matters for reproducibility: the subset drawn under `--limit`
    must be the same set on every machine and every run.
    """

    def __init__(self, root: str, transform=None):
        import pathlib

        self.root = pathlib.Path(root).expanduser()
        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} is not a directory")
        self.paths = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        )
        if not self.paths:
            raise FileNotFoundError(
                f"no images under {self.root} "
                f"(looked for {sorted(IMAGE_SUFFIXES)}, recursively)"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        # ImageNet's validation set contains a handful of CMYK and greyscale
        # JPEGs; convert unconditionally rather than discovering them at
        # batch 700 of a paid run.
        img = Image.open(self.paths[i]).convert("RGB")
        return (self.transform(img) if self.transform else img), 0


def build_loader(
    data: str | None,
    batch_size: int = 64,
    num_workers: int = 8,
    image_size: int = 224,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, str]:
    """Return (loader, source_tag).  `data=None` or "synthetic" uses fake images."""
    if data in (None, "synthetic"):
        ds = SyntheticImages(n=limit or 512, size=image_size, seed=seed)
        return (
            DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0),
            "synthetic",
        )

    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    # We never use labels -- every quantity here is computed from activation
    # covariances -- so there is no reason to demand the root/<class>/<image>
    # layout that ImageFolder wants.  A recursive scan accepts both that layout
    # and a flat directory, which means the ImageNet validation tar can be used
    # exactly as it unpacks, with no devkit and no valprep step.
    ds = FlatImageFolder(data, transform=tf)
    if limit is not None and limit < len(ds):
        # Deterministic subset, spread across classes rather than the first N
        # files (which would be a handful of classes and would bias every
        # covariance in the run).
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:limit].tolist()
        ds = torch.utils.data.Subset(ds, idx)

    return (
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        f"imagenet_val[{len(ds)}]",
    )
