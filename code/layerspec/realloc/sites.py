"""Sites: the architecture-specific part of width reallocation, reduced to a list.

A Site is one adjustable width: the module that produces it (its output channels or
features), the normalisation and activation that follow, and every module that consumes it.
Everything else in `layerspec.realloc` (measure, shrink, grow, allocate) is written against
this interface alone, so porting the controller to a new architecture means writing the
function that lists its Sites.  Three discoverers cover the common patterns:

    sequential_conv   Conv2d -> BatchNorm2d -> ReLU [-> MaxPool] -> next Conv2d / classifier
                      (VGG-style nn.Sequential bodies)
    resnet_internal   block-internal widths of BasicBlock / Bottleneck-shaped modules
                      (conv1 -> bn1 -> relu -> conv2 [-> bn2 -> relu -> conv3]); the block
                      output, which the shortcut ties to the residual stream, is left alone
    mlp               fc1 -> act -> fc2 (timm Mlp: transformers, ConvNeXt)

`discover_sites(model)` runs all three.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass(frozen=True)
class Consumer:
    """A module that reads the site's channels along its input axis.

    kind: "conv" (nn.Conv2d, in_channels) or "linear" (nn.Linear, in_features).
    const: where the constant of a shrink reconstruction goes: ("bias", <this module>) when
           the consumer has a bias, else ("bn_mean", <name of the BatchNorm right after it>).
    """
    name: str
    kind: str
    const: tuple[str, str]


@dataclass(frozen=True)
class Site:
    name: str                       # the producer's module name
    producer: str                   # nn.Conv2d or nn.Linear whose OUTPUT channels are the width
    norm: str | None                # BatchNorm2d / LayerNorm over those channels, or None
    act: tuple[str, bool]           # (module whose output is read, apply_relu): the post-activation tensor
    consumers: tuple[Consumer, ...]
    tie: str | None = None          # sites sharing a width (residual stream); v1 leaves tied sites alone
    extra: dict = field(default_factory=dict, compare=False)

    def width(self, model: nn.Module) -> int:
        m = dict(model.named_modules())[self.producer]
        return m.out_channels if isinstance(m, nn.Conv2d) else m.out_features

    def cost_per_channel(self, model: nn.Module) -> int:
        """Parameters that one channel of this site costs (producer row + norm + consumer columns)."""
        mods = dict(model.named_modules())
        p = mods[self.producer]
        if isinstance(p, nn.Conv2d):
            c = p.in_channels // p.groups * p.kernel_size[0] * p.kernel_size[1] + (1 if p.bias is not None else 0)
        else:
            c = p.in_features + (1 if p.bias is not None else 0)
        if self.norm is not None:
            c += 2
        for con in self.consumers:
            m = mods[con.name]
            if isinstance(m, nn.Conv2d):
                c += m.out_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
            else:
                c += m.out_features
        return c


# ----------------------------------------------------------------------------- discoverers

def _bn_after(seq: list[tuple[str, nn.Module]], i: int) -> str | None:
    for name, m in seq[i + 1:i + 3]:
        if isinstance(m, nn.BatchNorm2d):
            return name
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            break
    return None


def sequential_conv(model: nn.Module) -> list[Site]:
    """VGG-style: every Conv2d inside an nn.Sequential whose next Conv2d in the same Sequential
    consumes its output; the last one feeds the first nn.Linear with matching in_features."""
    sites = []
    mods = dict(model.named_modules())
    linears = [(n, m) for n, m in mods.items() if isinstance(m, nn.Linear)]
    for sname, seq in mods.items():
        if not isinstance(seq, nn.Sequential):
            continue
        items = [((sname + "." if sname else "") + n, m) for n, m in seq.named_children()]
        if any(isinstance(m, nn.Sequential) for _, m in items):
            continue                                   # go down to the flat Sequential
        convs = [i for i, (_, m) in enumerate(items) if isinstance(m, nn.Conv2d)]
        for j, i in enumerate(convs):
            cname, conv = items[i]
            if conv.groups != 1:
                continue
            bn = _bn_after(items, i)
            act = None
            for n, m in items[i + 1:i + 4]:
                if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU)):
                    act = (n, False)
                    break
                if isinstance(m, nn.Conv2d):
                    break
            if act is None:
                continue
            if j + 1 < len(convs):
                nname, nxt = items[convs[j + 1]]
                if nxt.groups != 1 or nxt.in_channels != conv.out_channels:
                    continue
                const = ("bias", nname) if nxt.bias is not None else ("bn_mean", _bn_after(items, convs[j + 1]) or "")
                if const[0] == "bn_mean" and not const[1]:
                    continue
                consumers = (Consumer(nname, "conv", const),)
            else:
                cand = [(n, m) for n, m in linears if m.in_features == conv.out_channels]
                if len(cand) != 1 or cand[0][1].bias is None:
                    continue
                consumers = (Consumer(cand[0][0], "linear", ("bias", cand[0][0])),)
            sites.append(Site(cname, cname, bn, act, consumers))
    return sites


def resnet_internal(model: nn.Module) -> list[Site]:
    """Block-internal widths of residual blocks shaped like torchvision's BasicBlock / Bottleneck
    (attributes conv1, bn1, conv2, bn2 [, conv3, bn3]).  The shared nn.ReLU is not hooked; the
    post-activation is read as relu(bn_k output)."""
    sites = []
    for bname, blk in model.named_modules():
        if not all(hasattr(blk, a) for a in ("conv1", "bn1", "conv2", "bn2")):
            continue
        if not (isinstance(blk.conv1, nn.Conv2d) and isinstance(blk.bn1, nn.BatchNorm2d)):
            continue
        p = bname + "." if bname else ""
        chain = [("conv1", "bn1", "conv2", "bn2")]
        if hasattr(blk, "conv3") and hasattr(blk, "bn3"):
            chain.append(("conv2", "bn2", "conv3", "bn3"))
        for prod, bn, nxt, nbn in chain:
            conv, cons = getattr(blk, prod), getattr(blk, nxt)
            if conv.groups != 1 or cons.groups != 1:
                continue
            const = ("bias", p + nxt) if cons.bias is not None else ("bn_mean", p + nbn)
            sites.append(Site(p + prod, p + prod, p + bn, (p + bn, True), (Consumer(p + nxt, "conv", const),)))
    return sites


def mlp(model: nn.Module) -> list[Site]:
    """timm-style Mlp modules: fc1 -> act -> fc2 (transformer blocks, ConvNeXt blocks)."""
    sites = []
    for bname, blk in model.named_modules():
        if not all(hasattr(blk, a) for a in ("fc1", "act", "fc2")):
            continue
        if not (isinstance(blk.fc1, nn.Linear) and isinstance(blk.fc2, nn.Linear)) or blk.fc2.bias is None:
            continue
        p = bname + "." if bname else ""
        sites.append(Site(p + "fc1", p + "fc1", None, (p + "act", False),
                          (Consumer(p + "fc2", "linear", ("bias", p + "fc2")),)))
    return sites


def discover_sites(model: nn.Module) -> list[Site]:
    """All sites the three discoverers find, in module order, without duplicates."""
    seen, out = set(), []
    for s in resnet_internal(model) + mlp(model) + sequential_conv(model):
        if s.producer not in seen:
            seen.add(s.producer)
            out.append(s)
    order = {n: i for i, (n, _) in enumerate(model.named_modules())}
    return sorted(out, key=lambda s: order.get(s.producer, 10**9))
