"""Five rulers on the same layers (pending I2': the "four kinds of dimension" supplement table).

For the 86 gate-passing dense convolutions of three trained ImageNet checkpoints (ResNet-18, ResNet-50, VGG-16_BN)
and the 33 of two random ones (plan 9.47 / 10.52), every ruler is read from the SAME stored conv-output spectrum
(results/local6400 or results/<model>_spectra.npz: 6,400 ImageNet-val images x 16 positions, sample-centred) or,
for the two TwoNN columns, from the same images (results/twonn/*_twonn.csv, measure_twonn.py):

    k*(0.95), k*(0.999)   principal components reaching tau of the variance (ours; Garg et al. use tau = 0.999)
    PR                    participation ratio (sum L)^2 / sum L^2 (Elmoznino & Bonner)
    erank                 exp(entropy of the normalised spectrum) (Roy & Vetterli)
    NR                    numerical rank: eigenvalues above 1e-3 of the largest (Masarczyk et al. 2023, two_rulers.py)
    ID(S1), ID(P)         TwoNN intrinsic dimension on position samples (one position per image, 6,400) and on the
                          pooled vectors (Ansuini et al.)

Nothing is re-measured: this script joins stored files.  It writes
    <out>/per_layer.csv      one row per layer, every ruler raw and divided by r_max
    <out>/summary.csv/.txt   per checkpoint: median level (ruler / r_max) and Spearman rho over layers against
                             k*(0.95) raw and against k*(0.95)/r_max
    <latex>                  the supplement table (rulers x checkpoints: level | rho against k*(0.95) raw)

    python scripts/ruler_table.py --results ../results --out ../results/rulers --latex ../paper/table_rulers.tex
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.metrics import effective_rank, k_star, participation_ratio  # noqa: E402

NR_THRESHOLD = 1e-3   # Masarczyk et al.: eigenvalues of the sample covariance above 1e-3 of the largest

CHECKPOINTS = [   # (twonn stem, label, trained?)
    ("timm_resnet18.tv_in1k", "ResNet-18", True),
    ("timm_resnet50.tv_in1k", "ResNet-50", True),
    ("timm_vgg16_bn.tv_in1k", "VGG-16\\_BN", True),
    ("timm_resnet18.tv_in1k_random", "ResNet-18 random", False),
    ("timm_vgg16_bn.tv_in1k_random", "VGG-16\\_BN random", False),
]
RULERS = ["k95", "k999", "PR", "erank", "NR", "id_S1", "id_P"]
RULER_TEX = {"k95": "$k^*(0.95)$", "k999": "$k^*(0.999)$", "PR": "PR", "erank": "erank", "NR": "NR ($10^{-3}$)",
             "id_S1": "TwoNN ID (positions)", "id_P": "TwoNN ID (pooled)"}


def numerical_rank(eig: np.ndarray, threshold: float = NR_THRESHOLD) -> int:
    eig = np.clip(np.asarray(eig, dtype=np.float64), 0, None)
    if eig.size == 0 or eig.max() <= 0:
        return 0
    return int((eig > threshold * eig.max()).sum())


def spectra_path(results: str, stem: str) -> str:
    """Same lookup as measure_twonn.stored_layers_csv, for the spectra."""
    if stem.endswith("_random"):
        return os.path.join(results, "local6400", "archs", "seed1", f"{stem}_spectra.npz")
    cand = os.path.join(results, "local6400", "archs", "trained", f"{stem}_spectra.npz")
    return cand if os.path.exists(cand) else os.path.join(results, f"{stem}_spectra.npz")


def per_layer(results: str, stem: str, label: str) -> pd.DataFrame:
    tw = pd.read_csv(os.path.join(results, "twonn", f"{stem}_twonn.csv"))
    spec = np.load(spectra_path(results, stem))
    rows = []
    for r in tw.to_dict("records"):
        key = f"{r['layer']}::conv"
        if key not in spec:
            raise KeyError(f"{stem}: no stored spectrum for {key}")
        eig = np.asarray(spec[key], dtype=np.float64)
        k95 = k_star(eig, 0.95)
        if abs(k95 - int(r["k95"])) > 2:
            raise ValueError(f"{stem} {r['layer']}: stored k*(0.95) {r['k95']} vs spectrum {k95}")
        rmax = float(r["r_max"])
        row = {"checkpoint": label, "stem": stem, "layer": r["layer"], "depth_index": int(r["depth_index"]),
               "C": int(r["C"]), "r_max": rmax,
               "k95": k95, "k999": k_star(eig, 0.999), "PR": participation_ratio(eig), "erank": effective_rank(eig),
               "NR": numerical_rank(eig), "id_S1": float(r["id_S1"]), "id_S1_sd": float(r["id_S1_sd"]),
               "id_P": float(r["id_P"]), "id_P_sd": float(r["id_P_sd"])}
        for c in RULERS:
            row[c + "_rmax"] = row[c] / rmax
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, g in df.groupby("checkpoint", sort=False):
        for c in RULERS:
            rho_raw = spearmanr(g[c], g["k95"]).correlation if c != "k95" else 1.0
            rho_norm = spearmanr(g[c + "_rmax"], g["k95_rmax"]).correlation if c != "k95" else 1.0
            out.append({"checkpoint": label, "ruler": c, "n_layers": len(g),
                        "level_rmax": float(g[c + "_rmax"].median()), "level_raw": float(g[c].median()),
                        "rho_raw": float(rho_raw), "rho_rmax": float(rho_norm),
                        "ratio_to_k95": float((g[c] / g["k95"]).median())})
    return pd.DataFrame(out)


def to_latex(summary: pd.DataFrame, labels: list[str]) -> str:
    lines = ["% Generated by scripts/ruler_table.py.  Do not edit by hand.",
             "\\begin{tabular}{l" + "rr" * len(labels) + "}", "\\toprule",
             "Ruler & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lb}}}" for lb in labels) + " \\\\",
             " & " + " & ".join("level & $\\rho$" for _ in labels) + " \\\\", "\\midrule"]
    for c in RULERS:
        cells = []
        for lb in labels:
            s = summary[(summary.checkpoint == lb) & (summary.ruler == c)].iloc[0]
            rho = "--" if c == "k95" else f"{s.rho_raw:.2f}".replace("-", "$-$")
            cells.append(f"{s.level_rmax:.2f} & {rho}")
        lines.append(f"{RULER_TEX[c]} & " + " & ".join(cells) + " \\\\")
    n = " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(summary[summary.checkpoint == lb].n_layers.iloc[0])}}}" for lb in labels)
    lines += ["\\midrule", f"layers & {n} \\\\", "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default="../results/rulers")
    p.add_argument("--latex", default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([per_layer(a.results, stem, label) for stem, label, _ in CHECKPOINTS], ignore_index=True)
    df.to_csv(os.path.join(a.out, "per_layer.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    labels = [lb for _, lb, _ in CHECKPOINTS]
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        fh.write("Five rulers on the same layers (ruler_table.py).  level = median over layers of ruler / r_max; "
                 "rho_raw = Spearman against k*(0.95) raw; rho_rmax = against k*(0.95)/r_max; "
                 "ratio = median of ruler / k*(0.95).\n\n")
        fh.write(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    print(open(os.path.join(a.out, "summary.txt")).read())
    if a.latex:
        with open(a.latex, "w") as fh:
            fh.write(to_latex(summary, labels))
        print(f"wrote {a.latex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
