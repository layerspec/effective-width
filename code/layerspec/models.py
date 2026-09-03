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


def build(name: str, pretrained: bool = True) -> tuple[nn.Module, str]:
    """Return (model, weights_tag).

    A name suffixed with ``_random`` gives the same architecture with default
    (untrained) initialisation.
    """
    import torchvision.models as tvm

    random_init = name.endswith("_random")
    base = name[: -len("_random")] if random_init else name

    if base not in REGISTRY:
        raise KeyError(f"unknown model {base!r}; known: {sorted(REGISTRY)}")

    ctor_name, weights_enum = REGISTRY[base]
    ctor = getattr(tvm, ctor_name)

    if random_init or not pretrained:
        return ctor(weights=None), "random_init"

    weights = getattr(tvm, weights_enum).IMAGENET1K_V1
    return ctor(weights=weights), str(weights)
