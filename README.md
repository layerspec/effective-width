# effective-width

Measurement code, results, and paper for

> **Measuring the Effective Width of Convolutional Layers: Protocol, Pitfalls,
> and the Comparison with Initialisation**

For every convolutional layer of a trained network, we ask how many directions
in channel space carry its activation variance, as a fraction of the directions
the layer can attain. The paper is a measurement protocol, three choices that
silently move published depth profiles, and a comparison of trained networks
with their own initialisation across seven architecture families.

## Findings in one paragraph

Across 19 public ImageNet checkpoints, (1) normalising by nominal width instead
of attainable rank caps every 1×1 expansion layer at its expansion ratio and
misreads bottleneck networks as underused; dividing by the attainable rank
removes the artefact in all 14 models that have such layers. (2) Estimating the
channel covariance after global average pooling gives a lower value than
sampling spatial positions on 687 of 703 dense layers. (3) The output of a
residual block exceeds the rank bound of its last convolution in 161 of 161
bottleneck blocks, so profiles read at block outputs and at convolution outputs
are different objects. With the protocol fixed, trained networks largely keep the layer ordering the
architecture gave them and gain a level, and the profile is in place before
accuracy is; a pre-registered controlled experiment on CIFAR-10 shows the block
type does not change this, so the one family whose public checkpoints reorder
their layers, ImageNet ResNet-50, owes it to its training regime.

## Reproducing the numbers

Every number in the paper is produced by a script from the released
per-layer eigenvalue spectra in `results/`; none is transcribed.

```bash
cd code
pip install -r requirements.txt            # torch only needed to re-measure

# Every statistic in Sections IV–V, plus the three LaTeX tables
python3 scripts/checkpoint_analysis.py --results ../results \
    --latex-models   ../paper/table_models.tex \
    --latex-pooling  ../paper/table_pooling.tex \
    --latex-families ../paper/table_families.tex > ../results/checkpoint_analysis.txt

# Figures 1–5
python3 -m layerspec.figures --results ../results --out ../results/figures
python3 scripts/paper_figures.py  --results ../results --out ../results/figures

# The paper (needs tectonic, or any TeX Live with IEEEtran)
cd ../paper && make tectonic
```

Re-measuring from scratch (one forward pass per checkpoint over the ImageNet
validation split; about five hours on one consumer GPU for all 22 runs):

```bash
python code/scripts/fetch_imagenet_val.py --out /data/imagenet_val   # needs `hf auth login`
bash   code/scripts/run_on_rented_gpu.sh /data/imagenet_val
```

The pipeline reads no labels, so a flat directory of validation images is
enough; no devkit or `valprep` reorganisation is needed. The cheaper second
measurement (six ResNet-50 recipes and five random seeds on a fixed
6,400-image subset; seven families with several checkpoints and seeds) runs on
Apple silicon: `code/scripts/run_local_seeds.sh`, `run_local_archs.sh`. The
CIFAR-10 validation gate and training trajectory are `reproduce_garg.py` and
`trajectory_cifar.py`.

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

## Layout

```
code/           the layerspec measurement package and the analysis scripts (see code/README.md)
results/        per-layer statistics (CSV), raw eigenvalue spectra (NPZ), run manifests,
                checkpoint_analysis.txt (the script output the paper quotes), figures/
paper/          main.tex, refs.bib, generated tables; `make tectonic` builds main.pdf
notes/          analysis plan with its deviation log (analysis-plan.md), the reviews and
                reading notes that shaped the paper (mostly in Chinese)
refs/           verification notes on contested references
```

## Pre-registration

`notes/analysis-plan.md` fixed the primary statistic, the sample-size gate and
the hypotheses before the ImageNet data were collected; its section 8 logs
every departure since, with the date and whether the data had been seen. The
shape hypotheses it registered did not survive a paired test over all layers,
and the paper says so. The git history of that file is the evidence.

## Status

Draft complete and compiling (2026-09-07); target venue IEEE TPAMI. The
repository is private until submission. Working notes in Chinese are in
`notes/status-zh.md`.
