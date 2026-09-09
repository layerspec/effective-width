"""Public API: profile(model, loader) and decompose(model, loader).

Both are thin wrappers over the measurement code the paper used
(hooks.SpectrumProbe, metrics, decomposition.PatchProbe); the paper's own
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

from . import decomposition as _dec
from . import metrics as _metrics
from .hooks import run_probe
from .metrics import DEFAULT_TAUS, tau_key as _tau_key


def select_device(device: str | None) -> str:
    """Explicit device, else cuda > mps > cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        the rows the paper's profiles are made of.  "Dense" here means
        ``kind == "conv" and not is_depthwise`` -- note ``decompose()``'s
        notion of dense is different (``groups == 1``): a grouped-but-not-
        depthwise convolution (ResNeXt) counts as dense here but is not
        decomposed there.

        Returns an empty frame with the usual columns (never raises) when the
        table itself is empty or lacks a ``kind`` column (e.g. a model whose
        Conv2d is never called in ``forward``, or an empty loader).  Raises
        ``ValueError`` if the table has rows but was never measured at `tau`."""
        col = f"k_star_rmax_{_tau_key(tau)}"
        t = self.table
        empty_cols = ["layer", "depth_index", "r_max", col]
        if t.empty or "kind" not in t.columns:
            return pd.DataFrame(columns=empty_cols)
        if col not in t.columns:
            available = sorted(float(c.rsplit("_", 1)[-1]) for c in t.columns
                               if c.startswith("k_star_rmax_"))
            raise ValueError(f"tau {tau} was not measured; available: {available}")
        keep = (t.kind == "conv") & t["ok"].astype(bool) & ~t["is_depthwise"].astype(bool)
        return (t.loc[keep, empty_cols]
                .sort_values("depth_index").reset_index(drop=True))

    def summary(self, tau: float = 0.95) -> dict:
        """L, median level, rho(depth) and the r_max sanity check, over
        `dense(tau)`.  Returns L=0 and NaNs (never raises) when the table is
        empty or lacks a `kind` column; raises ValueError, via `dense()`, if
        `tau` was never measured."""
        if self.table.empty or "kind" not in self.table.columns:
            return {"L": 0, "median_level": float("nan"), "rho_depth": float("nan"),
                    "rmax_check_max": float("nan"), "rmax_check_ok": True,
                    "rmax_check_tau": float("nan")}
        d = self.dense(tau)
        col = f"k_star_rmax_{_tau_key(tau)}"
        # r_max sanity check: use the *largest* available tau's k_star_rmax
        # column -- the check is tightest there -- rather than hardcoding
        # 0.999, since `taus=` may not include it.
        rmax_cols = [c for c in self.table.columns if c.startswith("k_star_rmax_")]
        conv = self.table.iloc[0:0]
        rmax_max = float("nan")
        rmax_check_tau = float("nan")
        if rmax_cols:
            check_col = max(rmax_cols, key=lambda c: float(c.split("_")[-1]))
            rmax_check_tau = float(check_col.rsplit("_", 1)[-1])
            conv = self.table[(self.table.kind == "conv") & self.table[check_col].notna()]
            rmax_max = float(conv[check_col].max()) if len(conv) else float("nan")
        rho = float("nan")
        if len(d) >= 3:
            rho = _spearman(np.arange(len(d)), d[col].to_numpy())
        return {"L": int(len(d)),
                "median_level": float(d[col].median()) if len(d) else float("nan"),
                "rho_depth": rho,
                "rmax_check_max": rmax_max,
                "rmax_check_ok": bool(rmax_max <= 1.0 + 1e-9) if len(conv) else True,
                "rmax_check_tau": rmax_check_tau}

    def warnings(self) -> list[str]:
        """Reproduces run.py's "!! N/M layers under-sampled" line for callers
        of the public API who never see that script's stdout: count of rows
        flagged `n_over_C_ok=False`, the worst layer (C and n/C), and the
        images needed at the profile's own `positions_per_image` to reach
        n/C = `min_n_over_C`.  Empty list when nothing is flagged."""
        t = self.table
        if t.empty or "n_over_C_ok" not in t.columns:
            return []
        bad = t[~t.n_over_C_ok]
        if bad.empty:
            return []
        worst = bad.loc[bad.n_over_C.idxmin()]
        positions = max(int(self.meta.get("positions_per_image", 1)), 1)
        min_n_over_C = self.meta.get("min_n_over_C", 50.0)
        need = int(np.ceil(min_n_over_C * worst.C / positions))
        return [f"!! {len(bad)}/{len(t)} layers under-sampled "
                f"(worst: {worst.layer}, C={int(worst.C)}, "
                f"n/C={worst.n_over_C:.1f}). "
                f"Need >= {need} images at --positions {positions}. "
                f"These rows are flagged n_over_C_ok=False -- do not report them."]

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

    `model` is moved to `device` for the duration of the measurement and left
    there afterward.  `model.training` is restored to its original value
    before returning (even if the measurement raises); the model is left in
    eval mode only while the forward passes run.
    """
    if not _has_conv(model):
        raise ValueError("profile() needs a model with at least one nn.Conv2d")
    device = select_device(device)
    was_training = model.training
    try:
        probe = run_probe(model, loader, device=device, positions_per_image=positions,
                          include_activations=include_activations, max_batches=max_batches,
                          seed=seed, progress=progress, pooled=True, include_blocks=include_blocks,
                          include_bn=include_bn)
    finally:
        model.train(was_training)
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
    convolution and split k*/r_max into out / kernel / data / ortho.

    "Dense" here means `groups == 1` -- note `Profile.dense()`'s notion of
    dense is different (`kind == "conv" and not is_depthwise`): a
    grouped-but-not-depthwise convolution (ResNeXt) is dense there but is not
    decomposed here.

    `model` is moved to `device` for the duration of the measurement and left
    there afterward.  `model.training` is restored to its original value
    before returning (even if the measurement raises).
    """
    if not any(isinstance(m, nn.Conv2d) and m.groups == 1 for m in model.modules()):
        raise ValueError("decompose() needs at least one dense (groups == 1) nn.Conv2d")
    device = select_device(device)
    was_training = model.training
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
        model.train(was_training)
    if not probe.acc:
        raise ValueError("no dense convolution accumulated a patch covariance: the forward pass "
                         "produced no multi-position windows (inputs too small?) or the loader was empty")
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
