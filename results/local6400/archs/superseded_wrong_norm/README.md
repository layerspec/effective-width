# Superseded measurements (wrong input normalisation)

The five weights in this directory were measured on 2026-09-07 with the ImageNet mean and standard deviation,
while they were trained with other input statistics (timm `pretrained_cfg`: 0.5 / 0.5, or 0 / 1 for the `miil`
weight). Under the wrong statistics their top-1 accuracy on 320 of the measurement images is 1.9-46.3%; under their
own it is 74.1-77.2% (`results/bplus_controls/norm_accuracy.txt`).

These files are kept as the record behind predictions P9.56 and P9.57 (analysis-plan 9.20 and 10.16) and are read only
by `scripts/bplus_controls_judge.py`. Every other analysis reads the corrected measurements, which were copied from
`results/bplus_controls/renorm/` into `results/local6400/archs/trained/` on 2026-09-30.
