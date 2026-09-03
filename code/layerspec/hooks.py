"""Instrument a torchvision model and accumulate per-layer channel covariance.

Design notes
------------
*What counts as a layer.*  We hook every ``nn.Conv2d`` output and, separately,
every activation (``ReLU``/``GELU``/``SiLU``) output.  These are different
objects and the distinction matters: the conv output is what that layer's
filters actually produced, while the post-activation is what the next layer
sees.  ReLU makes the representation non-negative, which changes the covariance
structure, so mixing the two in one curve would be a confound.  Both are
recorded and tagged in ``kind``.

*What counts as a sample.*  One spatial position of one image.  We take
``positions_per_image`` positions drawn uniformly at random per image per layer
rather than all H*W, because neighbouring positions are strongly correlated:
counting them all inflates the nominal sample size without adding independent
information, which would make the finite-sampling control (control.py) look
better than it is.

*Where the arithmetic happens.*  The centred block scatter is formed on the
accelerator in float64 when the device supports it at reasonable speed, else in
float32 after per-block centring (which keeps magnitudes small enough that
float32 is adequate for one block).  Only the C-vector and C x C matrix cross
back to the host, where the running accumulator is always float64.

*Depthwise convolutions.*  Flagged via ``groups``.  A depthwise conv does not
mix channels, so its channel covariance means something different from a dense
conv's; ConvNeXt is mostly depthwise and this must be visible in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .accumulate import CovarianceAccumulator

ACTIVATIONS = (nn.ReLU, nn.GELU, nn.SiLU, nn.Hardswish, nn.ELU)


@dataclass
class LayerRecord:
    name: str
    kind: str                 # "conv" | "act"
    depth_index: int          # order of first execution
    C: int
    groups: int = 1
    is_depthwise: bool = False
    # Which invocation of this module within one forward pass this record is.
    # ResNet blocks reuse a single nn.ReLU for every activation in the block,
    # so the module NAME does not identify an activation site: a Bottleneck
    # calls self.relu three times, at 64, 64 and 256 channels.  Keying by name
    # alone merges all three into one accumulator -- which crashes on the
    # shape change in ResNet-50, and silently pools two different
    # distributions in ResNet-18, where the shapes happen to match.
    call_index: int = 0
    spatial: tuple[int, int] | None = None
    acc: CovarianceAccumulator | None = field(default=None, repr=False)
    # Second covariance over globally average-pooled feature maps: one sample
    # per image instead of one per spatial position.  This is the estimator
    # Elmoznino & Bonner (2024) used, and it is a genuinely different object --
    # pooling averages the spatial variance away.  Carrying both lets the paper
    # test whether the disagreement in the literature is the pooling and not
    # the metric.  Note it is inherently sample-limited: n = number of images,
    # so a 2048-channel layer over 50k images only reaches n/C ~ 24.
    acc_pooled: CovarianceAccumulator | None = field(default=None, repr=False)
    # n -> metrics row, recorded as the sample count crosses each threshold.
    checkpoints: dict = field(default_factory=dict, repr=False)
    _next_ckpt: int = 0

    def meta(self) -> dict:
        return {
            "layer": self.name,
            "kind": self.kind,
            "depth_index": self.depth_index,
            "C": self.C,
            "groups": self.groups,
            "is_depthwise": self.is_depthwise,
            "call_index": self.call_index,
            "H": self.spatial[0] if self.spatial else None,
            "W": self.spatial[1] if self.spatial else None,
        }


class SpectrumProbe:
    """Attach to a model, run batches through it, collect per-layer covariance."""

    def __init__(
        self,
        model: nn.Module,
        positions_per_image: int = 16,
        include_activations: bool = True,
        max_channels: int | None = None,
        seed: int = 0,
        scatter_dtype: torch.dtype | None = None,
        checkpoint_multiples: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100, 250),
        pooled: bool = True,
    ):
        self.model = model
        self.positions_per_image = int(positions_per_image)
        self.include_activations = include_activations
        self.max_channels = max_channels
        self.scatter_dtype = scatter_dtype
        self.pooled = pooled
        # Thresholds are multiples of C, per layer: a 64-channel layer needs far
        # fewer samples than a 2048-channel one, and the honest x-axis for the
        # convergence control is n/C, not n.
        self.checkpoint_multiples = tuple(sorted(checkpoint_multiples))
        self.layers: dict[str, LayerRecord] = {}
        self._handles: list = []
        self._order = 0
        # Reset at the start of every forward pass so a module invoked N times
        # gets N stable, distinct records rather than one merged one.
        self._calls: dict[str, int] = {}
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self._handles.append(
            model.register_forward_pre_hook(lambda _m, _i: self._calls.clear())
        )
        self._attach()

    # ---------------------------------------------------------------- attach

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                groups = module.groups
                self._register(
                    name,
                    module,
                    kind="conv",
                    groups=groups,
                    is_depthwise=(groups > 1 and groups == module.in_channels),
                )
            elif self.include_activations and isinstance(module, ACTIVATIONS):
                self._register(name, module, kind="act")

    def _register(self, name: str, module: nn.Module, kind: str, **extra) -> None:
        base = f"{name}::{kind}"

        def hook(_mod, _inp, out, _base=base, _name=name, _kind=kind, _extra=extra):
            if not isinstance(out, torch.Tensor) or out.dim() != 4:
                return
            idx = self._calls.get(_base, 0)
            self._calls[_base] = idx + 1
            self._consume(f"{_base}#{idx}", _name, _kind, idx,
                          out.detach(), _extra)

        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # --------------------------------------------------------------- consume

    def _consume(self, key, name, kind, call_index, out: torch.Tensor,
                 extra: dict) -> None:
        N, C, H, W = out.shape
        if self.max_channels is not None and C > self.max_channels:
            return

        rec = self.layers.get(key)
        if rec is None:
            rec = LayerRecord(
                name=name,
                kind=kind,
                depth_index=self._order,
                C=C,
                groups=extra.get("groups", 1),
                is_depthwise=extra.get("is_depthwise", False),
                spatial=(H, W),
                call_index=call_index,
                acc=CovarianceAccumulator(C),
                acc_pooled=CovarianceAccumulator(C) if self.pooled else None,
            )
            self.layers[key] = rec
            self._order += 1

        work_dtype = self.scatter_dtype or (
            torch.float64 if out.device.type == "cpu" else torch.float32
        )

        x = self._sample_positions(out, N, C, H, W).to(work_dtype)   # (m, C)
        self._fold(rec.acc, x)
        self._maybe_checkpoint(rec)

        if rec.acc_pooled is not None:
            # (N, C): one sample per image, spatial extent averaged away.
            self._fold(rec.acc_pooled, out.mean(dim=(2, 3)).to(work_dtype))

    @staticmethod
    def _fold(acc: CovarianceAccumulator, x: torch.Tensor) -> None:
        block_mean = x.mean(dim=0)
        xc = x - block_mean
        acc.update_precomputed(
            x.shape[0],
            block_mean.double().cpu().numpy(),
            (xc.T @ xc).double().cpu().numpy(),
        )

    def _maybe_checkpoint(self, rec: LayerRecord) -> None:
        """Record metrics whenever this layer's sample count crosses n = k*C.

        Done inline so the whole convergence control costs one pass over the
        data instead of one pass per sample size.
        """
        from . import metrics as _metrics

        while rec._next_ckpt < len(self.checkpoint_multiples):
            mult = self.checkpoint_multiples[rec._next_ckpt]
            if rec.acc.n < mult * rec.C:
                return
            if rec.acc.n >= 2:
                row = _metrics.compute(
                    rec.acc.eigenvalues(), rec.C, rec.acc.n
                ).to_row()
                row["n_over_C"] = mult
                rec.checkpoints[mult] = row
            rec._next_ckpt += 1

    def _sample_positions(self, out, N, C, H, W) -> torch.Tensor:
        """Return (N * p, C) with p random spatial positions drawn per image."""
        hw = H * W
        p = min(self.positions_per_image, hw)
        flat = out.permute(0, 2, 3, 1).reshape(N, hw, C)   # (N, HW, C)
        if p == hw:
            return flat.reshape(N * hw, C)
        idx = torch.randint(0, hw, (N, p), generator=self._gen)   # cpu generator
        idx = idx.to(flat.device)
        gathered = torch.gather(flat, 1, idx.unsqueeze(-1).expand(N, p, C))
        return gathered.reshape(N * p, C)

    # ---------------------------------------------------------------- output

    def records(self) -> list[LayerRecord]:
        return sorted(self.layers.values(), key=lambda r: r.depth_index)

    def spectra(self) -> dict[str, np.ndarray]:
        return {
            f"{r.name}::{r.kind}#{r.call_index}": r.acc.eigenvalues()
            for r in self.records()
            if r.acc is not None and r.acc.n >= 2
        }


@torch.no_grad()
def run_probe(
    model: nn.Module,
    loader,
    device: str = "cuda",
    positions_per_image: int = 16,
    include_activations: bool = True,
    max_batches: int | None = None,
    seed: int = 0,
    progress: bool = True,
    pooled: bool = True,
) -> SpectrumProbe:
    """Push batches from `loader` through `model` with a probe attached."""
    model = model.to(device).eval()
    probe = SpectrumProbe(
        model,
        positions_per_image=positions_per_image,
        include_activations=include_activations,
        seed=seed,
        pooled=pooled,
    )
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(images.to(device, non_blocking=True))
            if progress and (i % 20 == 0):
                print(f"  batch {i}", flush=True)
    finally:
        probe.remove()
    return probe
