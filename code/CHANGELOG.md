# Changelog

## 0.2.0 (2026-09-09)
- The ruler on `nn.Linear`: `r_max = min(in, out)`, token sequences (N, T, D) sampled over tokens like spatial positions; `Profile.dense(kinds=("linear",))`, `summary(kinds=...)`; `decompose(include_linear=True)` with the exact identity on all tokens. Transformers (timm/torchvision ViT) profile out of the box; residual blocks of transformers are not hooked as `block` yet.
- Public API: `layerspec.profile()`, `layerspec.decompose()`, `Profile`, `Decomposition`,
  `load_model()`, `image_loader()`, `select_device()`.
- `layerspec` console script with `profile` and `decompose` subcommands.
- The decomposition core moved from `scripts/decompose_check.py` into `layerspec.decompose`;
  the script is now a wrapper and reproduces the paper's ResNet-50 table row for row.
- Packaging via `pyproject.toml`; torchvision/timm are optional (`[models]`).

## 0.1.0
- Measurement code as used for the paper's rounds 1-3.
