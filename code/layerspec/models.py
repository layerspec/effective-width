"""Model registry.

The set is chosen to answer specific questions, not for breadth:

  vgg16_bn        the architecture Garg, Panda & Roy (2019) measured, so our
                  numbers can be checked against theirs before we trust the
                  rest.  Plain, no skip connections.
  resnet18/50     Garg explicitly EXCLUDED ResNets over shortcut connections.
                  This is the gap.
  convnext_tiny   postdates every paper in this literature; mostly depthwise,
                  LayerNorm rather than BatchNorm.  Second gap.
  densenet121     dense connectivity, a third connectivity pattern.

  *_random        the same architecture at initialisation, untrained.  This is
                  the control that separates "structure the network learned"
                  from "structure that follows from the architecture and the
                  input statistics alone".  It costs one extra pass and no
                  paper in the scan reported it.

  timm:<name>     any timm checkpoint, e.g. ``timm:resnet50.a1_in1k``.  Added
                  2026-09-03 to fix n = 1: every architecture above had one
                  set of public weights and therefore no error bar.  The
                  tiers below are the checkpoint plan in
                  notes/checkpoints-2026-09-03.md; ``tier1``/``tier2`` on the
                  command line expand to them.
"""

from __future__ import annotations

import torch.nn as nn

REGISTRY = {
    "vgg16_bn": ("vgg16_bn", "VGG16_BN_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "densenet121": ("densenet121", "DenseNet121_Weights"),
}

DEFAULT_MODELS = ("vgg16_bn", "resnet18", "resnet50", "convnext_tiny")

# Same architecture, different training recipes: r_max is identical across
# the six, so any spread is training and not architecture.  Then four
# bottleneck families whose expansion ratio -- and hence r_max/C at the
# bound layers -- differs from ResNet-50's 0.25: if k*/r_max holds its level
# while k*/C tracks r_max/C, the denominator correction is doing what it
# claims.
TIER1 = (
    "timm:resnet50.tv_in1k",
    "timm:resnet50.tv2_in1k",
    "timm:resnet50.a1_in1k",
    "timm:resnet50.a3_in1k",
    "timm:resnet50.gluon_in1k",
    "timm:resnet50.fb_ssl_yfcc100m_ft_in1k",
    "timm:resnet101.tv_in1k",          # r_max/C = 0.25 at 33 expansions
    "timm:wide_resnet50_2.tv_in1k",    # 0.5
    "timm:resnext50_32x4d.tv_in1k",    # 0.5; grouped 3x3 not binding
    "timm:mobilenetv2_100.ra_in1k",    # 1/6 at 16 expansions
)
TIER2 = (
    "timm:efficientnet_b0.ra_in1k",
    "timm:mobilenetv3_large_100.ra_in1k",
    "timm:vgg16.tv_in1k",              # VGG-16 without BN
    "timm:resnet34.a1_in1k",           # basic blocks: almost no bound layers
    "timm:convnext_tiny.fb_in22k_ft_in1k",
    "timm:densenet121.tv_in1k",
)
TIERS = {"tier1": TIER1, "tier2": TIER2}


def expand_names(names) -> list[str]:
    """Replace ``tier1``/``tier2`` tokens with their checkpoint lists."""
    out: list[str] = []
    for n in names:
        out.extend(TIERS.get(n, (n,)))
    return out


def parse(name: str) -> tuple[str, str, bool]:
    """Return (backend, base_name, random_init) for a model name."""
    random_init = name.endswith("_random")
    base = name[: -len("_random")] if random_init else name
    if base.startswith("timm:"):
        return "timm", base[len("timm:"):], random_init
    if base.startswith("tvdet:"):
        return "tvdet", base[len("tvdet:"):], random_init
    return "torchvision", base, random_init


def file_stem(name: str) -> str:
    """Model name as it appears in output file names (no ':' on any FS)."""
    return name.replace(":", "_")


def build(name: str, pretrained: bool = True, seed: int = 0) -> tuple[nn.Module, str]:
    """Return (model, weights_tag).

    A name suffixed with ``_random`` gives the same architecture with default
    (untrained) initialisation, drawn from ``seed`` so that the control is the
    same weights on every run.  (The 2026-09-03 run did not seed it, and its
    resnet50_random differs from the first run's; see the test.)
    """
    import torch

    backend, base, random_init = parse(name)
    if random_init or not pretrained:
        torch.manual_seed(seed)

    if backend == "tvdet":
        # The ResNet body of a torchvision detection model (COCO-trained, FrozenBatchNorm);
        # ablation A11.  The forward returns a dict of feature maps; only the hooks matter.
        import torchvision.models.detection as tvd
        ctor = getattr(tvd, base)
        m = ctor(weights=None) if (random_init or not pretrained) else ctor(weights="DEFAULT")
        return m.backbone.body, (f"random_init(seed={seed})" if random_init else f"torchvision/{base}.DEFAULT (backbone.body)")
    if backend == "timm":
        import timm

        if random_init or not pretrained:
            return timm.create_model(base, pretrained=False), f"random_init(seed={seed})"
        return timm.create_model(base, pretrained=True), f"timm/{base}"

    import torchvision.models as tvm

    if base not in REGISTRY:
        raise KeyError(f"unknown model {base!r}; known: {sorted(REGISTRY)}")

    ctor_name, weights_enum = REGISTRY[base]
    ctor = getattr(tvm, ctor_name)

    if random_init or not pretrained:
        return ctor(weights=None), f"random_init(seed={seed})"

    weights = getattr(tvm, weights_enum).IMAGENET1K_V1
    return ctor(weights=weights), str(weights)
