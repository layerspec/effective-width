"""layerspec -- per-layer effective dimensionality of trained CNNs.

Measures, for every conv layer of a pretrained network, how many principal
components of its channel-space activation covariance are actually used,
relative to the layer's nominal channel count.

The claim this supports: Garg, Panda & Roy (IEEE Access 2019) measured this for
VGG on CIFAR and found a hunchback; Elmoznino & Bonner (PLOS CB 2024) measured
a related quantity (participation ratio) across many ImageNet models and found
a monotonic rise.  Nobody has published the *normalised* k*/C curve for ResNet
or ConvNeXt at ImageNet scale, and nobody has computed both metrics on the same
layers of the same models -- which is what would reconcile the two results.
"""

__version__ = "0.2.0"

from .accumulate import CovarianceAccumulator, merge  # noqa: F401
from . import metrics  # noqa: F401
from .api import (profile, Profile, decompose, Decomposition, select_device,  # noqa: F401
                  load_model, image_loader)
