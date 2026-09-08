# layerspec 0.2: a public API for the paper's measurements

Date: 2026-09-09. Status: approved in conversation, implementation pending.
Submission-bar item I1.

## Goal

Anyone with a PyTorch CNN and a DataLoader can obtain, in a few lines, the
per-layer effective-width profile the paper defines (k*(τ)/r_max with the
n/C gate) and its decomposition into kernel and data factors. The paper's
own scripts call the same functions, so the released numbers and the tool
cannot diverge.

## Scope

In: `profile()`, `decompose()`, result objects, model/loader conveniences,
a thin CLI, packaging (`pyproject.toml`, PyPI-ready), tests, README.
Out (v0.2, later): projection calibration (stays a script), figures,
creating the PyPI account (author does it; commands are provided).

## Public interface

```python
import layerspec
model  = layerspec.load_model("resnet50")                 # or "timm:convnext_tiny", or any nn.Module
loader = layerspec.image_loader("/data/imagenet_val_6400")  # or any DataLoader yielding tensors or (x, y)

prof = layerspec.profile(model, loader, device=None, positions=16,
                         taus=(0.9, 0.95, 0.99, 0.999), min_n_over_C=50,
                         include_activations=True, include_blocks=True,
                         include_bn=False, max_batches=None, seed=0)
prof.table          # DataFrame, one row per hooked tensor: layer, kind (conv/act/block/bn),
                    #   depth_index, C, groups, is_depthwise, r_max, n_samples, n_over_C, ok,
                    #   k_star_<tau>, k_star_rmax_<tau>, participation_ratio, effective_rank,
                    #   stable_rank, powerlaw_alpha (same columns as results/*_layers.csv)
prof.dense(tau=0.95)  # conv rows, not depthwise, ok == True: layer, depth_index, k_star_rmax_<tau>
prof.summary(tau=0.95)  # dict: L, median level, rho_depth, max k*(0.999)/r_max (must be <= 1)
prof.spectra        # {layer: eigenvalues}
prof.to_csv(path); layerspec.Profile.from_csv(path)

dec = layerspec.decompose(model, loader, device=None, positions=48,
                          taus=(0.9, 0.95, 0.99, 0.999), rank_tol=1e-6,
                          max_batches=None, seed=0)
dec.table           # one row per dense conv: layer, depth_index, C_in, C_out, k, d, r_max,
                    #   n_samples, n_over_d, kernel_erank_over_rmax, kernel_cond_top_rmax,
                    #   data_erank_over_rmax, out_<tau>, kernel_<tau>, data_<tau>, ortho_<tau>,
                    #   ostrowski_ok (same columns as results/decompose/*_decompose.csv)
dec.check()         # dict: Ostrowski count, corollary counts per tau, kappa median/max
```

Behaviour:
- `device=None` picks cuda, then mps, then cpu.
- A model with no `nn.Conv2d` raises `ValueError`.
- Layers failing the gate are flagged (`ok=False`), never dropped; `dense()`
  applies the gate.
- Batches may be tensors or `(x, y, ...)` tuples; the first element is used.
- `r_max = groups * min((C_in/groups) * k_h * k_w, C_out/groups)` (paper rule 3).
- Depthwise convolutions are flagged and excluded from `dense()` (rule 5).

## Files

- `code/pyproject.toml` (new): name `layerspec`, version 0.2.0, Python >= 3.10;
  dependencies torch >= 2.0, numpy, pandas; extras `models` (torchvision, timm),
  `dev` (pytest). Build backend: setuptools. Console script `layerspec`.
- `code/layerspec/api.py` (new): `profile`, `decompose`, `Profile`, `Decomposition`,
  `load_model`, `image_loader`, device selection.
- `code/layerspec/decompose.py` (new): the Σ_patch accumulation and the four
  quantities, moved from `scripts/decompose_check.py`.
- `code/layerspec/cli.py` (new): `layerspec profile|decompose --model --data --out [...]`.
- `code/layerspec/__init__.py`: export the public names; version bump.
- `code/scripts/decompose_check.py`: becomes a thin wrapper over the package.
- `code/layerspec/run.py`: unchanged (the paper's batch runner).
- `code/tests/test_api.py` (new).
- `README.md` and `code/README.md`: "Use it on your own model" section.
- `paper/main.tex` Reproducibility: one sentence, `pip install layerspec`.

## Tests

CPU, seconds:
- `profile()` on a small CNN (dense + depthwise + 1x1 expansion) with random
  images: expected columns present; k*(0.999)/r_max <= 1 on every row;
  depthwise flagged; gate flag matches n/C; CSV round-trip equal.
- `decompose()` on the same network: `out` recomputed from W and Σ_patch equals
  the directly measured k*/r_max within 0.01; `ortho >= out` at every tau;
  Ostrowski ratios within bounds; `check()` counts consistent.
- Existing `tests/test_core.py` unchanged and passing.

Regression (run once, not in the suite): `decompose()` on ResNet-50 over the
6,400-image subset must reproduce `results/decompose/resnet50_decompose.csv`
row for row.

## Non-goals

No new measurement semantics. No change to the numbers in the paper.
