# layerspec

Per-layer effective width of trained convolutional networks: the measurement
behind *Measuring the Effective Width of Convolutional Layers* (see the
repository README for the paper and the findings).

For every convolutional layer of a network, `layerspec` accumulates the
channel-space covariance of the layer's output (at the convolution output,
before the nonlinearity and before any residual addition; post-activation and
block outputs are recorded separately), from sampled spatial positions and,
in parallel, from globally pooled features. From each spectrum it reports
$k^*(\tau)$, participation ratio, effective rank and stable rank, each divided
by the nominal channel count $C$ and by the attainable rank
$r_{\max} = g\cdot\min((C_{\rm in}/g)\,k_h k_w,\ C_{\rm out}/g)$, together
with the sample-to-channel ratio at which each statistic was estimated and a
convergence curve in $n/C$.

**No training is needed** to reproduce the paper's ImageNet measurements:
everything runs on downloaded pretrained weights, one forward pass per
checkpoint. The CIFAR-10 validation gate and the training trajectory are the
only scripts that train anything.

## Use it on your own model

[![PyPI](https://img.shields.io/pypi/v/layerspec.svg)](https://pypi.org/project/layerspec/)

```bash
pip install layerspec            # torch, numpy, pandas
pip install "layerspec[models]"  # + torchvision, timm for the named checkpoints
```

```python
import layerspec

model  = layerspec.load_model("resnet50")                  # or any torch.nn.Module with Conv2d layers
loader = layerspec.image_loader("/path/to/6400/images")    # or any DataLoader yielding images or (x, y)

prof = layerspec.profile(model, loader)     # one forward pass; 16 positions per image
prof.dense(tau=0.95)                        # layer, depth_index, r_max, k*(0.95)/r_max  (gated, dense convs only)
prof.summary()                              # L, median level, rho(depth), max k*(0.999)/r_max (must be <= 1)
prof.to_csv("resnet50_layers.csv")

dec = layerspec.decompose(model, loader)    # out / kernel / data / ortho per dense conv
dec.table[["layer", "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95"]]
dec.check()                                 # Proposition-1 counts
```

Or from the shell: `layerspec profile --model resnet50 --data /path/to/images --out resnet50.csv`. Linear layers are measured too (`r_max = min(in, out)`, tokens as positions), so a ViT profiles out of the box: `layerspec profile --model timm:vit_base_patch16_224.augreg_in21k_ft_in1k --positions 32 ...`; read them with `prof.dense(kinds=("linear",))`.

What the numbers mean: `k*(tau)/r_max` is the number of principal directions of a
convolution's output covariance (before the nonlinearity) carrying a fraction
`tau` of its variance, divided by the rank the layer can attain
(`groups * min(C_in/groups * kh * kw, C_out/groups)`). Rows with `ok == False`
have fewer than 50 samples per channel and are not reportable. Depthwise
convolutions are flagged and left out of `dense()`. A `k*(0.999)/r_max` above 1
means `r_max` is wrong for that layer type; please report it. Grouped
convolutions that are not depthwise (ResNeXt) appear in `profile().dense()`
but are not decomposed by `decompose()`, which only handles `groups == 1`.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Smoke test — no data needed, verifies the pipeline runs (2 minutes, CPU is fine)
python -m layerspec.run --models resnet18_random --data synthetic \
    --limit 96 --batch-size 16 --image-size 96 --device cpu --out /tmp/smoke

# 2. Correctness tests
PYTHONPATH=. python tests/test_core.py

# 3. Validation gate — reproduce Garg et al. Table 2 (~1 GPU-hour)
python scripts/reproduce_garg.py --epochs 100 --out results/garg

# 4. The real run
python -m layerspec.run \
    --data /path/to/imagenet/val \
    --models vgg16_bn resnet18 resnet50 convnext_tiny \
             resnet50_random convnext_tiny_random \
    --out results/

# 5. Paper figures
python -m layerspec.figures --results results/ --out results/figures
```

Or all of it in one go on a rented box:

```bash
bash scripts/run_on_rented_gpu.sh /path/to/imagenet/val
```

## The validation gate

`scripts/reproduce_garg.py` trains VGG-16_BN on CIFAR-10 and compares our
per-layer k\*(0.999) against Garg et al.'s published Table 2. **Run this before
the ImageNet pass.** If the shape does not reproduce, the problem is our
pipeline, not the literature, and every downstream number is suspect.

The gate is about shape, not digits — their training recipe is not fully
specified, and k\* is sensitive to how well the network converged:

| outcome | meaning |
|---|---|
| r > 0.9, MAD < 0.10 | reproduced — proceed |
| r > 0.7 | broadly consistent — investigate the layers that disagree |
| r < 0.7 | **stop** — something in the pipeline or the training is wrong |

ImageNet validation must be laid out as `root/<class>/<image>.JPEG` — the
standard `ImageFolder` layout. It requires registration at image-net.org; the
pipeline does not download it for you.

---

## The one rule that matters

**A sample covariance built from n < C samples is rank-deficient by construction.**
Its k\* is an artefact of the sample size, not a property of the network. This is
one of the three biases Pospisil & Pillow (PNAS 2025) identify in the V1
eigenspectrum literature, and it will silently manufacture exactly the result
you are hoping to see.

So:

- Every row carries `n_over_C` and a boolean `n_over_C_ok` (default threshold
  n/C ≥ 50). **Rows with `n_over_C_ok=False` must not be reported.** The runner
  prints a warning and tells you how many images you need.
- `<model>_convergence.csv` gives the metrics recomputed at n/C = 1, 2, 5, 10, 25,
  50, 100, 250 for every layer, at no extra cost (one pass). **A layer whose k\*/C
  has not flattened by the largest available n/C is not a measurement.** Plot this
  before plotting anything else.

With the full 50,000-image validation set at `--positions 16`, the deepest
ResNet-50 layer (C = 2048) gets n/C ≈ 390. Comfortable.

## The other thing that will get caught in review

Spatially adjacent activations are strongly correlated, so N·H·W is **not** the
number of independent samples. Counting all positions inflates the nominal
sample size without adding information — it makes the convergence control look
better than it is. This pipeline therefore samples `--positions` random spatial
positions per image per layer (default 16) and reports that as n.

---

## Outputs

| file | contents |
|---|---|
| `<model>_layers.csv` | one row per layer: metadata + all metrics at full n |
| `<model>_convergence.csv` | one row per (layer, n/C) checkpoint — the sample-size control |
| `<model>_spectra.npz` | raw eigenvalues per layer, so any metric can be recomputed without another pass |
| `manifest.json` | run settings, timings, versions |
| `figures/fig1_profile` | k\*/C vs depth, per architecture, trained vs random-init |
| `figures/fig2_metrics` | k\*/C vs PR/C vs effective-rank/C on the same layers — the reconciliation panel |
| `figures/fig3_convergence_*` | the sample-size control |
| `figures/fig4_threshold_*` | τ sensitivity |

Figures are written as both PDF (for the paper) and PNG (for looking at), sized
for IEEE two-column, with every series carrying a distinct marker and dash
pattern as well as a colour so they survive greyscale print.

### Metrics, and why there are four

Four different quantities get called "the dimensionality of a representation"
and they are **not** interchangeable — Ansuini et al. (2019) report PC-ID ≈ 200
and TwoNN ID ≈ 18 for the same VGG-16 layer. Everything here is a *linear*
measure from the covariance spectrum; none of it estimates manifold dimension.
Keep the names straight in the paper.

- `k_star_{0.9,0.95,0.99,0.999}` — PCs reaching τ of total variance. Garg et al.
  used τ = 0.999, which is looser than 0.95 and inflates the ratio; report the
  threshold sensitivity.
- `participation_ratio` — (Σλ)²/Σλ². The Elmoznino & Bonner quantity.
  Threshold-free, but head-dominated.
- `effective_rank` — exp(entropy of the normalised spectrum), Roy & Vetterli (2007).
  Threshold-free and responds to the tail.
- `stable_rank` — Σλ/λ₁. Cheapest, most head-dominated.
- `powerlaw_alpha` — slope of log λ vs log i over `[powerlaw_fit_lo, powerlaw_fit_hi]`.
  **Always report the window**: Kong et al. (2022) fit PCs 10–999, and the value
  moves a lot with it. Kong et al. found α ≪ 1 for standard CNNs — which is why
  Marchenko–Pastur thresholding is unusable here: no bulk edge to cut at.

All are reported raw and divided by C.

### Layers

`kind` distinguishes `conv` (the conv module's own output) from `act` (post-ReLU
/ GELU). These are different objects — ReLU makes the representation
non-negative, which changes the covariance structure — so mixing them in one
curve is a confound. `is_depthwise` flags depthwise convs, which do not mix
channels; ConvNeXt is mostly depthwise and this has to be visible.

### The `_random` control

`resnet50_random` is the same architecture at initialisation. This separates
*structure the network learned* from *structure that follows from the
architecture and input statistics alone*. It costs one extra pass, and no paper
in the literature scan reported it.

---

## Width reallocation: `layerspec.realloc`

The measurement above reads a width off a trained layer. `layerspec.realloc`
acts on it: remove the channels the covariance says are redundant and fold
them into the layers that consume them (no retraining), add channels along the
input directions the layer does not yet span (the function is unchanged), and
split a channel budget across layers from their spectra. Everything is written
against one interface, so porting to a new architecture is one function.

### The interface: a `Site`

A `Site` is one adjustable width: the module that produces it, the
normalisation and activation that follow, and every module that reads it
(`Consumer`s, with the axis they read along and where a folded constant goes:
the consumer's bias, or the running mean of the next BatchNorm). Three
discoverers cover the common patterns; `discover_sites(model)` runs all three.

| Discoverer | Pattern | Example |
|---|---|---|
| `sequential_conv` | `Conv2d → BatchNorm2d → ReLU [→ MaxPool] → next Conv2d / classifier` inside an `nn.Sequential` | VGG |
| `resnet_internal` | the widths *inside* a `BasicBlock` / `Bottleneck`-shaped module (`conv1 → bn1 → relu → conv2 [→ bn2 → relu → conv3]`); the block output, tied to the residual stream by the shortcut, is left alone | ResNet-18/50, ResNet-20 |
| `mlp` | `fc1 → act → fc2` (timm `Mlp`) | ViT, DeiT, ConvNeXt |

A site the discoverers do not find is a `Site(...)` you construct yourself.

### The operators

```python
from layerspec.realloc import discover_sites, measure, greedy_cssp, shrink, grow, water_fill

sites = discover_sites(model)                       # list[Site]
mom   = measure(model, sites, calib_loader, device) # post-activation covariance and input-patch
                                                    # covariance per site; inputs only, labels ignored
s = sites[5]
order, traces = greedy_cssp(mom[s.name].act_cov)    # channels in the order a pivoted Cholesky keeps them,
                                                    # with the residual variance after each
shrink(model, s, order[:m], mom[s.name])            # keep m channels; the rest are reconstructed
                                                    # linearly and folded into every consumer
grow(model, s, k, mom[s.name])                      # k new channels along the input directions the
                                                    # site does not span; consumers get zero rows,
                                                    # so the output is identical to the last bit
widths, mu, _ = water_fill(spectra, cost, budget)   # one width per site at a total cost, by the
                                                    # KKT conditions of the summed spectra
```

`removable_width(traces, eps)` is the smallest number of channels whose
reconstruction residual is at most `eps` of the variance — the ruler of the
paper read from the subset rather than the eigenvalues. `set_module` keeps the
replaced module's train/eval mode (a fresh `BatchNorm` defaults to training and
would silently switch to batch statistics).

### What it has been checked on

`scripts/realloc_check.py --device mps` runs the package end to end on the
paper's VGG-16 (BN) CIFAR-10 network (seed 0, full width 93.83 %):

* `discover_sites` finds the 13 sites.
* Shrinking every site to the post-activation `k*(0.999)` widths and folding
  the rest, with no retraining, gives 93.82 % at 67.3 % of the parameters —
  the same network, parameter for parameter, as the paper's fold script.
* Growing every site by 8 channels along its unmet directions leaves the
  accuracy at 93.83 % exactly; the stem gains nothing, because a 3×3 RGB patch
  spans at most 27 directions and the 64 filters already cover them.
* Water-filling the same parameter budget as the ruler widths chooses a
  different allocation (early layers full, the last three cut deeper) at the
  same 93.82 %.

`tests/test_realloc.py` covers discovery on VGG / ResNet-20 / `Mlp`,
bit-exact function preservation under `grow`, exact removal of a dead channel,
equality of the folded shrink with the hooked reconstruction, the two-sided
CSSP bounds, and the KKT and budget conditions of `water_fill`.

### What it does not do

* It does not decide *whether* a network has width to remove. On CIFAR-10 at
  ten classes the ruler frees a third of VGG-16's parameters; on CIFAR-100 and
  ImageNet-1k, and in a ten-class ImageNet ResNet-18, the same ruler reads
  the full width (a ten-class ImageNet VGG-16 keeps 0.52–0.62 of its
  parameters), and folding removes nothing. Measure first.
* It does not train. The in-training controller (shrink and grow on a
  schedule) lives in `scripts/realloc_controller.py` and
  `scripts/trajectory_cifar.py --realloc-budget`; on CIFAR-10 its gain over
  a uniform width at equal FLOPs is confined to VGG-16, and on ResNet-20 and a
  CIFAR ViT it equals the uniform baseline. It is not part of the package API.
* `shrink` folds a *linear* reconstruction of the dropped post-activation
  channels into the consumers. The fold itself is exact up to zero-padding
  borders and any pooling between the activation and the consumer; what is
  approximate is the reconstruction, whose residual variance is `traces[m]`,
  and the accuracy cost is what the check above measures.

---

## Cost

| | |
|---|---|
| Models | 4 pretrained + 2 random-init controls |
| Data | ImageNet val, 50,000 images, one pass each |
| Time | ~20–40 min per model on one RTX 3090 |
| Total | ~30–50 GPU-hours including reruns and debugging |
| Rented | RTX 3090 at ≈ US$0.09/hr → **under NT$500** |

This is pure inference. A 4090 buys nothing here; a 3090 or even a 3060 is fine.
