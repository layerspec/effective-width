"""Public API: profile(model, loader) and decompose(model, loader).

Both are thin wrappers over the measurement code the paper used
(hooks.SpectrumProbe, metrics, decompose.PatchProbe); the paper's own
scripts call the same functions, so the released numbers and this tool
cannot diverge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import metrics as _metrics
from .hooks import run_probe

DEFAULT_TAUS = (0.9, 0.95, 0.99, 0.999)


def select_device(device: str | None) -> str:
    """Explicit device, else cuda > mps > cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tau_key(tau: float) -> str:
    return f"{tau:g}"


def record_row(rec, min_n_over_C: float, taus: tuple[float, ...] = DEFAULT_TAUS) -> dict | None:
    """One table row for a hooked tensor, or None if it has too few samples to
    form a covariance at all.  This is the row layout of results/*_layers.csv."""
    if rec.acc is None or rec.acc.n < 2:
        return None
    eig = rec.acc.eigenvalues()
    row = rec.meta()
    row.update(_metrics.compute(eig, rec.C, rec.acc.n, taus=taus).to_row())
    row["n_over_C"] = rec.acc.n / rec.C
    # A sample covariance from n < C samples is rank-deficient by construction;
    # below n/C = 50 the small eigenvalues are still biased.  Flag, never drop.
    row["n_over_C_ok"] = bool(rec.acc.n >= min_n_over_C * rec.C)
    if rec.r_max:
        for tau in taus:
            k = row.get(f"k_star_{_tau_key(tau)}")
            if k is not None:
                row[f"k_star_rmax_{_tau_key(tau)}"] = k / rec.r_max
        row["r_max_over_C"] = rec.r_max / rec.C
    if rec.acc_pooled is not None and rec.acc_pooled.n >= 2:
        pm = _metrics.compute(rec.acc_pooled.eigenvalues(), rec.C, rec.acc_pooled.n, taus=taus).to_row()
        for k, v in pm.items():
            if k != "C":
                row[f"pooled_{k}"] = v
        row["pooled_n_over_C"] = rec.acc_pooled.n / rec.C
        row["pooled_n_over_C_ok"] = bool(rec.acc_pooled.n >= min_n_over_C * rec.C)
    return row


def _spearman(a, b) -> float:
    a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Profile:
    """Per-layer effective-width profile of one network on one image set."""
    table: pd.DataFrame
    spectra: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def dense(self, tau: float = 0.95) -> pd.DataFrame:
        """Dense convolutions that pass the sample-size gate, in depth order:
        the rows the paper's profiles are made of."""
        col = f"k_star_rmax_{_tau_key(tau)}"
        t = self.table
        keep = (t.kind == "conv") & t["ok"].astype(bool) & ~t["is_depthwise"].astype(bool)
        return (t.loc[keep, ["layer", "depth_index", "r_max", col]]
                .sort_values("depth_index").reset_index(drop=True))

    def summary(self, tau: float = 0.95) -> dict:
        d = self.dense(tau)
        col = f"k_star_rmax_{_tau_key(tau)}"
        # r_max sanity check: use the *largest* available tau's k_star_rmax
        # column -- the check is tightest there -- rather than hardcoding
        # 0.999, since `taus=` may not include it.
        rmax_cols = [c for c in self.table.columns if c.startswith("k_star_rmax_")]
        conv = self.table.iloc[0:0]
        rmax_max = float("nan")
        if rmax_cols:
            check_col = max(rmax_cols, key=lambda c: float(c.split("_")[-1]))
            conv = self.table[(self.table.kind == "conv") & self.table[check_col].notna()]
            rmax_max = float(conv[check_col].max()) if len(conv) else float("nan")
        rho = float("nan")
        if len(d) >= 3:
            rho = _spearman(np.arange(len(d)), d[col].to_numpy())
        return {"L": int(len(d)),
                "median_level": float(d[col].median()) if len(d) else float("nan"),
                "rho_depth": rho,
                "rmax_check_max": rmax_max,
                "rmax_check_ok": bool(rmax_max <= 1.0 + 1e-9) if len(conv) else True}

    def to_csv(self, path) -> None:
        self.table.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path) -> "Profile":
        # dtype=str for layer: nn.Sequential module names are digit strings
        # ("0", "1", ...), which pandas' CSV sniffer otherwise reads back as
        # int64, breaking equality against the in-memory (str) table.
        t = pd.read_csv(path, dtype={"layer": str})
        if "ok" not in t.columns and "n_over_C_ok" in t.columns:
            t["ok"] = t["n_over_C_ok"].astype(bool)
        return cls(table=t, meta={"source": os.fspath(path)})


def _has_conv(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


@torch.no_grad()
def profile(model: nn.Module, loader, *, device: str | None = None, positions: int = 16,
            taus: tuple[float, ...] = DEFAULT_TAUS, min_n_over_C: float = 50.0,
            include_activations: bool = True, include_blocks: bool = True, include_bn: bool = False,
            max_batches: int | None = None, seed: int = 0, progress: bool = False) -> Profile:
    """Measure k*(tau)/r_max (and PR, effective rank, stable rank) for every
    convolution output of `model` over the images `loader` yields.

    `loader` may yield tensors or (x, y, ...) tuples; the first element is the
    image batch.  Every hooked tensor gets a row; `Profile.dense()` applies the
    paper's gate and depthwise exclusion.
    """
    if not _has_conv(model):
        raise ValueError("profile() needs a model with at least one nn.Conv2d")
    device = select_device(device)
    probe = run_probe(model, loader, device=device, positions_per_image=positions,
                      include_activations=include_activations, max_batches=max_batches,
                      seed=seed, progress=progress, pooled=True, include_blocks=include_blocks,
                      include_bn=include_bn)
    rows, spectra = [], {}
    for rec in probe.records():
        row = record_row(rec, min_n_over_C, taus)
        if row is None:
            continue
        row["ok"] = row["n_over_C_ok"]
        rows.append(row)
        spectra[f"{rec.name}::{rec.kind}#{rec.call_index}"] = rec.acc.eigenvalues()
    table = pd.DataFrame(rows)
    meta = {"device": device, "positions_per_image": positions, "seed": seed,
            "min_n_over_C": min_n_over_C, "torch": torch.__version__, "n_rows": len(table)}
    return Profile(table=table, spectra=spectra, meta=meta)


from . import decompose as _dec


@dataclass
class Decomposition:
    """Kernel/data decomposition of the effective width, one row per dense convolution."""
    table: pd.DataFrame
    patch_covariance: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def check(self) -> dict:
        return _dec.check_rows(self.table, tuple(self.meta.get("taus", _dec.DEFAULT_TAUS)))

    def to_csv(self, path) -> None:
        self.table.to_csv(path, index=False)


@torch.no_grad()
def decompose(model: nn.Module, loader, *, device: str | None = None, positions: int = 48,
              taus: tuple[float, ...] = DEFAULT_TAUS, rank_tol: float = 1e-6,
              max_batches: int | None = None, seed: int = 0) -> Decomposition:
    """Accumulate the receptive-field patch covariance of every dense
    convolution and split k*/r_max into out / kernel / data / ortho."""
    if not any(isinstance(m, nn.Conv2d) and m.groups == 1 for m in model.modules()):
        raise ValueError("decompose() needs at least one dense (groups == 1) nn.Conv2d")
    device = select_device(device)
    model = model.to(device).eval()
    probe = _dec.PatchProbe(model, positions, seed)
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))
    finally:
        probe.remove()
    rows = _dec.decompose_rows(probe, taus, rank_tol)
    table = pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)
    cov = {name: acc.covariance for name, acc in probe.acc.items()}
    meta = {"device": device, "positions": positions, "seed": seed, "rank_tol": rank_tol,
            "taus": tuple(taus), "torch": torch.__version__}
    return Decomposition(table=table, patch_covariance=cov, meta=meta)


def load_model(name: str, *, pretrained: bool = True, seed: int = 0) -> nn.Module:
    """A torchvision model by registry name ("resnet50", "vgg16_bn", ...), a timm
    checkpoint ("timm:resnet50.a1_in1k"), or either with the suffix "_random"
    for the same architecture at initialisation.  Needs the [models] extra."""
    from .models import build
    model, _tag = build(name, pretrained=pretrained, seed=seed)
    return model


def image_loader(path: str | None, *, batch_size: int = 32, workers: int = 4, image_size: int = 224,
                 limit: int | None = None, seed: int = 0, preprocess: str = "crop"):
    """A DataLoader over every image under `path` (any directory layout; labels
    are not needed), with ImageNet normalisation.  `path=None` or "synthetic"
    gives 1/f-noise images for smoke tests.  Needs the [models] extra for real images."""
    from .data import build_loader
    loader, _src = build_loader(path, batch_size=batch_size, num_workers=workers, image_size=image_size,
                                limit=limit, seed=seed, preprocess=preprocess)
    return loader
