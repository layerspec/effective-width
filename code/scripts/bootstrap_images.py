"""Block bootstrap over images for the paper's headline statistics (D1 / M3).

Every number in the paper is a point estimate on one image sample.  The six
ResNet-50 recipes bound the spread across *weights*; nothing bounds the spread
across *images*.  This script closes that gap without storing activations:

  collect    one forward pass over the image set on the CPU, in image blocks
             of `--block-images`; for every conv / block-output site it saves
             the block's (n, mean, centred scatter) -- exactly the quantities
             the streaming accumulator merges (Chan et al. 1983) -- and, for
             every dense conv, the block's mean and variance of the patch
             input projected on the kernel's retained right-singular
             directions (all that rho_align needs).
  bootstrap  resamples blocks with replacement `--reps` times, merges the
             saved block statistics into a covariance for every replicate,
             and recomputes k*(tau), erank, PR, pooled versions and
             rho_align.  Replicate 0 is every block once: the point estimate.
  summarise  turns replicates into percentile CIs for the statistics the
             paper reports (pooling count and drop, r_max effect size,
             block-vs-conv3 count, trained-vs-random rho, rho_align median,
             per-layer k*/C half-widths) and writes the LaTeX table.

Blocks are consecutive images of the sorted directory.  A block bootstrap is
the right unit here because the per-layer statistic is a functional of the
whole covariance, not a mean of per-image quantities; resampling images one
at a time would need a covariance per image.  With B = 32 blocks of 200
images the CI is coarse but honest; nothing below assumes normality.

The sample-size gate is applied exactly as in the main pipeline (n/C >= 50),
so a site whose gate can never pass at this image count is not even stored.
On 6,400 images the pooled estimator (one sample per image) passes the gate
only for C <= 128; the positional estimator (16 per image) for C <= 2048.

Usage (from code/):
    python scripts/bootstrap_images.py collect   --model resnet50 --data ../data/imagenet_val_6400 --out ../results/bootstrap
    python scripts/bootstrap_images.py bootstrap --model resnet50 --out ../results/bootstrap --reps 200
    python scripts/bootstrap_images.py summarise --out ../results/bootstrap --trained resnet50 vgg16_bn --random resnet50_random --latex ../paper/table_bootstrap.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Thread caps must precede the numpy import: this runs beside a training
# queue on the same machine and must not take every core.
if "--threads" in sys.argv:
    _t = sys.argv[sys.argv.index("--threads") + 1]
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_k, _t)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAUS = (0.9, 0.95, 0.99, 0.999)
PRIMARY = "k_star_ratio_0.95"
K999 = "k_star_ratio_0.999"


# ------------------------------------------------------------------ core maths

def draw_counts(n_blocks: int, reps: int, seed: int) -> np.ndarray:
    """(reps + 1, B) multiplicities.  Row 0 is every block once (the point
    estimate); rows 1..reps are draws with replacement."""
    rng = np.random.default_rng(seed)
    out = np.ones((reps + 1, n_blocks), dtype=np.int64)
    for r in range(1, reps + 1):
        out[r] = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
    return out


def merge_blocks(counts: np.ndarray, n_b: np.ndarray, mean_b: np.ndarray, M2_b) -> tuple[int, np.ndarray]:
    """Merge blocks with multiplicities `counts` into (n, covariance).

    Law of total scatter: M2 = sum_b c_b M2_b + sum_b c_b n_b (mu_b - mu)(mu_b - mu)^T,
    mu the count-weighted mean.  `M2_b` may be a float32 memmap of shape
    (B, C, C); the sum is formed in float64.
    """
    counts = np.asarray(counts, dtype=np.int64)
    w = counts * np.asarray(n_b, dtype=np.float64)
    n = float(w.sum())
    if n < 2:
        raise ValueError("fewer than two samples after merging")
    mean_b = np.asarray(mean_b, dtype=np.float64)
    mu = (w[:, None] * mean_b).sum(0) / n
    C = mean_b.shape[1]
    M2 = np.zeros((C, C), dtype=np.float64)
    for b in np.nonzero(counts)[0]:
        M2 += counts[b] * np.asarray(M2_b[b], dtype=np.float64)
        d = mean_b[b] - mu
        M2 += w[b] * np.outer(d, d)
    cov = M2 / (n - 1.0)
    return int(n), (cov + cov.T) * 0.5


def merge_diag(counts: np.ndarray, n_b: np.ndarray, mean_b: np.ndarray, M2d_b: np.ndarray) -> tuple[int, np.ndarray]:
    """Same identity restricted to the diagonal: per-direction variances."""
    counts = np.asarray(counts, dtype=np.float64)
    w = counts * np.asarray(n_b, dtype=np.float64)
    n = float(w.sum())
    mu = (w[:, None] * mean_b).sum(0) / n
    var = ((counts[:, None] * M2d_b).sum(0) + (w[:, None] * (mean_b - mu) ** 2).sum(0)) / (n - 1.0)
    return int(n), var


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) <= 1e-6 * np.abs(a).max() or np.ptp(b) <= 1e-6 * np.abs(b).max():
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def jitter(v: np.ndarray) -> float:
    return float(np.median(np.abs(np.diff(v)))) if len(v) > 1 else float("nan")


def site_metrics(eig: np.ndarray, C: int, n: int, r_max) -> dict:
    from layerspec import metrics as _metrics
    row = _metrics.compute(eig, C, n, taus=TAUS).to_row()
    for tau in TAUS:
        k = _metrics.tau_key(tau)
        row[f"k_star_rmax_{k}"] = row[f"k_star_{k}"] / r_max if r_max else float("nan")
    return row


# ------------------------------------------------------------------- collect

class _DiagAcc:
    """Streaming mean and per-coordinate scatter (diagonal Chan merge)."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def update(self, z: np.ndarray) -> None:
        m = z.shape[0]
        if m == 0:
            return
        bm = z.mean(0)
        bM2 = ((z - bm) ** 2).sum(0)
        if self.n == 0:
            self.n, self.mean, self.M2 = m, bm, bM2
            return
        na, nb = self.n, m
        d = bm - self.mean
        self.M2 = self.M2 + bM2 + d * d * (na * nb / (na + nb))
        self.mean = self.mean + d * (nb / (na + nb))
        self.n = na + nb


class ProjectionProbe:
    """Per dense conv: patches projected on the r retained right-singular
    directions of W, accumulated per block as mean and variance.  rho_align
    only needs v_i^T Sigma_patch v_i, i.e. these r variances."""

    def __init__(self, model, positions: int, seed: int, rank_tol: float = 1e-6):
        import torch
        import torch.nn as nn
        self.p = positions
        self.gen = torch.Generator().manual_seed(seed)
        self.sites: dict[str, dict] = {}
        self.acc: dict[str, _DiagAcc] = {}
        self.handles = []
        order = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                W = m.weight.detach().double().reshape(m.out_channels, -1).cpu().numpy()
                _, sv, Vt = np.linalg.svd(W, full_matrices=False)
                r = int((sv > sv[0] * rank_tol).sum())
                self.sites[name] = dict(depth_index=order, C_in=m.in_channels, C_out=m.out_channels,
                                        d=W.shape[1], k=m.kernel_size[0] * m.kernel_size[1],
                                        r=r, sv=sv[:r], V=torch.from_numpy(Vt[:r].T.astype(np.float32)))
                order += 1
                self.handles.append(m.register_forward_pre_hook(
                    lambda mod, inp, _n=name: self._consume(_n, mod, inp[0])))

    def _consume(self, name, m, x) -> None:
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            V = self.sites[name]["V"]
            zs = []
            # per image, so the im2col tensor never exceeds one image's worth
            for i in range(x.shape[0]):
                cols = F.unfold(x[i:i + 1], m.kernel_size, dilation=m.dilation,
                                padding=m.padding, stride=m.stride)[0]      # (d, L)
                L = cols.shape[1]
                if L == 1:
                    return
                p = min(self.p, L)
                idx = torch.randint(0, L, (p,), generator=self.gen)
                zs.append(cols[:, idx].T.float() @ V)                       # (p, r)
            z = torch.cat(zs, 0).double().numpy()
        acc = self.acc.get(name)
        if acc is None:
            acc = self.acc[name] = _DiagAcc(V.shape[1])
        acc.update(z)

    def take(self) -> dict[str, _DiagAcc]:
        out, self.acc = self.acc, {}
        return out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def collect(args) -> None:
    import torch
    from layerspec import data as _data, models as _models
    from layerspec.accumulate import CovarianceAccumulator
    from layerspec.hooks import SpectrumProbe

    torch.set_num_threads(args.threads)
    out = os.path.join(args.out, args.model)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "meta.json")) and not args.force:
        print(f"{out}/meta.json exists, skipping collect (use --force)")
        return

    model, tag = _models.build(args.model, seed=args.seed)
    model = model.to("cpu").eval()
    loader, source = _data.build_loader(args.data, batch_size=args.batch_size, num_workers=args.workers,
                                        image_size=args.image_size, limit=args.limit, seed=args.seed)
    N = len(loader.dataset)
    if args.block_images % args.batch_size:
        raise SystemExit("--block-images must be a multiple of --batch-size")
    B = N // args.block_images
    if B < 2:
        raise SystemExit(f"only {N} images: fewer than two blocks of {args.block_images}")
    n_used = B * args.block_images
    print(f"{args.model} ({tag}) on {source}: {B} blocks x {args.block_images} images "
          f"({N - n_used} left over), positions {args.positions}, threads {args.threads}", flush=True)

    probe = SpectrumProbe(model, positions_per_image=args.positions, include_activations=False,
                          include_linear=False, seed=args.seed, checkpoint_multiples=(),
                          pooled=True, include_blocks=True)
    proj = ProjectionProbe(model, positions=args.patch_positions, seed=args.seed + 1)

    sites: dict[str, dict] = {}
    arrays: dict[str, dict] = {}
    proj_store: dict[str, dict] = {}

    def flush(b: int) -> None:
        for rec in probe.records():
            key = f"{rec.name}::{rec.kind}#{rec.call_index}"
            for which, acc in (("pos", rec.acc), ("pooled", rec.acc_pooled)):
                if acc is None or acc.n == 0:
                    continue
                if acc.n * B / rec.C < args.min_n_over_C:
                    continue                                   # gate can never pass: do not store
                if key not in sites:
                    sites[key] = dict(rec.meta(), key=key)
                    sites[key]["r_max"] = None if rec.r_max is None else int(rec.r_max)
                if which not in arrays.setdefault(key, {}):
                    fname = f"{'' if which == 'pos' else 'pooled_'}M2_{list(sites).index(key)}.npy"
                    arrays[key][which] = dict(
                        M2=np.lib.format.open_memmap(os.path.join(out, fname), mode="w+",
                                                     dtype=np.float32, shape=(B, rec.C, rec.C)),
                        mean=np.zeros((B, rec.C)), n=np.zeros(B, dtype=np.int64))
                    sites[key][f"{which}_file"] = os.path.basename(arrays[key][which]["M2"].filename)
                a = arrays[key][which]
                a["M2"][b] = acc.M2.astype(np.float32)
                a["mean"][b] = acc.mean
                a["n"][b] = acc.n
            rec.acc = CovarianceAccumulator(rec.C)
            if rec.acc_pooled is not None:
                rec.acc_pooled = CovarianceAccumulator(rec.C)
        for name, acc in proj.take().items():
            s = proj_store.setdefault(name, dict(mean=np.zeros((B, acc.mean.shape[0])),
                                                 M2=np.zeros((B, acc.mean.shape[0])),
                                                 n=np.zeros(B, dtype=np.int64)))
            s["mean"][b], s["M2"][b], s["n"][b] = acc.mean, acc.M2, acc.n

    t0 = time.time()
    b, seen = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x)
            seen += x.shape[0]
            if seen == args.block_images:
                flush(b)
                b += 1
                seen = 0
                print(f"  block {b}/{B}  {time.time() - t0:.0f}s", flush=True)
                if b == B:
                    break
    probe.remove()
    proj.remove()

    for key, d in arrays.items():
        for which, a in d.items():
            a["M2"].flush()
            base = a["M2"].filename[:-4]
            np.save(base.replace("M2_", "mean_") + ".npy", a["mean"])
            np.save(base.replace("M2_", "n_") + ".npy", a["n"])
    np.savez(os.path.join(out, "proj.npz"),
             **{f"{k}::{f}": v for k, s in proj_store.items() for f, v in s.items()},
             **{f"{k}::sv": proj.sites[k]["sv"] for k in proj_store})
    proj_meta = {k: {f: v for f, v in proj.sites[k].items() if f not in ("V", "sv")} for k in proj_store}
    meta = dict(model=args.model, weights=tag, source=source, n_blocks=B, block_images=args.block_images,
                positions=args.positions, patch_positions=args.patch_positions, seed=args.seed,
                min_n_over_C=args.min_n_over_C, elapsed_sec=round(time.time() - t0, 1),
                torch=torch.__version__, sites=list(sites.values()), proj_sites=proj_meta)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {len(sites)} sites, {len(proj_store)} projection sites, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------- bootstrap

def _load_site(out: str, fname: str):
    M2 = np.load(os.path.join(out, fname), mmap_mode="r")
    mean = np.load(os.path.join(out, fname.replace("M2_", "mean_")))
    n = np.load(os.path.join(out, fname.replace("M2_", "n_")))
    return n, mean, M2


def bootstrap(args) -> None:
    out = os.path.join(args.out, args.model)
    meta = json.load(open(os.path.join(out, "meta.json")))
    B = meta["n_blocks"]
    counts = draw_counts(B, args.reps, args.seed)
    dest = os.path.join(out, "replicates.csv")
    if os.path.exists(dest) and not args.force:
        print(f"{dest} exists, skipping (use --force)")
        return
    t0 = time.time()
    rows = []
    for si, s in enumerate(meta["sites"]):
        parts = {}
        for which in ("pos", "pooled"):
            if f"{which}_file" not in s:
                continue
            n_b, mean_b, M2_b = _load_site(out, s[f"{which}_file"])
            per_rep = []
            for r in range(counts.shape[0]):
                n, cov = merge_blocks(counts[r], n_b, mean_b, M2_b)
                eig = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
                m = site_metrics(eig, s["C"], n, s["r_max"])
                m["n_samples"] = n
                m["n_over_C"] = n / s["C"]
                per_rep.append(m)
            parts[which] = per_rep
        for r in range(counts.shape[0]):
            row = dict(rep=r, layer=s["layer"], kind=s["kind"], depth_index=s["depth_index"], C=s["C"],
                       r_max=s["r_max"], groups=s["groups"], is_depthwise=s["is_depthwise"])
            if "pos" in parts:
                row.update(parts["pos"][r])
            if "pooled" in parts:
                row.update({f"pooled_{k}": v for k, v in parts["pooled"][r].items()})
            rows.append(row)
        print(f"  site {si + 1}/{len(meta['sites'])} {s['layer']} C={s['C']}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["n_over_C_ok"] = df["n_over_C"] >= meta["min_n_over_C"] if "n_over_C" in df else False
    if "pooled_n_over_C" in df:
        df["pooled_n_over_C_ok"] = df["pooled_n_over_C"].fillna(0) >= meta["min_n_over_C"]
    df.to_csv(dest, index=False)

    # rho_align per replicate
    pz = np.load(os.path.join(out, "proj.npz"))
    arows = []
    for name, pm in meta["proj_sites"].items():
        n_b, mean_b, M2_b, sv = pz[f"{name}::n"], pz[f"{name}::mean"], pz[f"{name}::M2"], pz[f"{name}::sv"]
        gain = sv ** 2
        for r in range(counts.shape[0]):
            n, var = merge_diag(counts[r], n_b, mean_b, M2_b)
            arows.append(dict(rep=r, layer=name, depth_index=pm["depth_index"], C_out=pm["C_out"], d=pm["d"],
                              rank_W=pm["r"], n_samples=n, rho_align=spearman(gain, var)))
    pd.DataFrame(arows).to_csv(os.path.join(out, "align.csv"), index=False)
    print(f"done: {len(df)} rows, {len(arows)} align rows, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------- summarise

def _ci(v, lo=2.5, hi=97.5) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if len(v) else (float("nan"),) * 2


def _dense_conv(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.kind == "conv") & df.n_over_C_ok.astype(bool)]
    return d[~d.is_depthwise.astype(bool)]


def per_rep_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per replicate: the statistics the paper reports for one model."""
    out = []
    for r, g in df.groupby("rep"):
        d = _dense_conv(g).sort_values("depth_index")
        row = dict(rep=r, n_layers=len(d))
        # pooling (rule: pooled gate too)
        if "pooled_n_over_C_ok" in d:
            p = d[d.pooled_n_over_C_ok.astype(bool)]
            delta = (p["pooled_" + PRIMARY] - p[PRIMARY]).to_numpy()
            s_r = (p["k_star_0.95"] / p["r_max"]).to_numpy()
            p_r = (p["pooled_k_star_0.95"] / p["r_max"]).to_numpy()
            row.update(pool_n=len(p), pool_lower=int((delta < 0).sum()),
                       pool_drop_c=-float(np.median(delta)) if len(p) else np.nan,
                       pool_jit_c=jitter(p[PRIMARY].to_numpy()),
                       pool_drop_r=-float(np.median(p_r - s_r)) if len(p) else np.nan,
                       pool_jit_r=jitter(s_r))
            row["pool_ratio_c"] = row["pool_drop_c"] / row["pool_jit_c"]
            row["pool_ratio_r"] = row["pool_drop_r"] / row["pool_jit_r"]
        # r_max: bounded vs unbounded, tau = 0.999
        bd, un = d[d.r_max < d.C], d[d.r_max >= d.C]
        if len(bd) >= 3 and len(un) >= 3:
            from scipy.stats import mannwhitneyu
            row.update(rmax_nb=len(bd), rmax_nu=len(un),
                       rmax_diff_c=float(bd[K999].median() - un[K999].median()),
                       rmax_diff_r=float(bd["k_star_rmax_0.999"].median() - un["k_star_rmax_0.999"].median()),
                       rmax_p_c=float(mannwhitneyu(bd[K999], un[K999]).pvalue),
                       rmax_p_r=float(mannwhitneyu(bd["k_star_rmax_0.999"], un["k_star_rmax_0.999"]).pvalue))
        row["rmax_max"] = float(d["k_star_rmax_0.999"].max()) if len(d) else np.nan
        # block output vs conv3 (bottleneck ResNets)
        blocks = g[(g.kind == "block") & g.n_over_C_ok.astype(bool)].set_index("layer")
        c3 = d[d.layer.str.endswith(".conv3")].copy()
        if len(blocks) and len(c3):
            c3["block"] = c3.layer.str[: -len(".conv3")]
            c3 = c3.set_index("block")
            common = c3.index.intersection(blocks.index)
            if len(common):
                bound = (c3.loc[common, "r_max"] / c3.loc[common, "C"]).to_numpy()
                kc, kb = c3.loc[common, K999].to_numpy(), blocks.loc[common, K999].to_numpy()
                row.update(block_n=len(common), block_over=int((kb > bound + 1e-9).sum()),
                           block_ratio=float(np.median(kb / kc)))
        # level and depth trend
        row["level_c"] = float(d[PRIMARY].median())
        row["level_r"] = float(d["k_star_rmax_0.95"].median())
        row["rho_depth"] = spearman(np.arange(len(d)), d[PRIMARY].to_numpy())
        out.append(row)
    return pd.DataFrame(out).set_index("rep")


def per_layer_halfwidths(df: pd.DataFrame, cols) -> dict:
    d = _dense_conv(df)
    out = {}
    for c in cols:
        if c not in d:
            continue
        hw = []
        for layer, g in d.groupby("layer"):
            v = g[g.rep > 0][c].to_numpy()
            if len(v) and not np.isnan(v).all():
                lo, hi = _ci(v)
                hw.append((hi - lo) / 2)
        out[c] = (float(np.median(hw)), float(np.max(hw))) if hw else (np.nan, np.nan)
    return out


def trained_vs_random(t: pd.DataFrame, r: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for rep in sorted(t.rep.unique()):
        a = _dense_conv(t[t.rep == rep]).sort_values("depth_index").set_index("layer")
        b = _dense_conv(r[r.rep == rep]).sort_values("depth_index").set_index("layer")
        common = a.index.intersection(b.index)
        row = dict(rep=rep, n=len(common))
        for c in cols:
            row[f"tr_{c}"] = spearman(a.loc[common, c].to_numpy(), b.loc[common, c].to_numpy())
            row[f"tr_meddiff_{c}"] = float(np.median(a.loc[common, c].to_numpy() - b.loc[common, c].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("rep")


def align_stats(al: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep, g in al.groupby("rep"):
        v = g.rho_align.dropna().to_numpy()
        rows.append(dict(rep=rep, n=len(v), median=float(np.median(v)) if len(v) else np.nan,
                         neg=int((v < 0).sum()), pos=int((v > 0).sum())))
    return pd.DataFrame(rows).set_index("rep")


def _fmt(point, ci, nd=3, integer=False):
    if integer:
        return f"{int(point)} [{int(round(ci[0]))}, {int(round(ci[1]))}]"
    return f"{point:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def summarise(args) -> None:
    lines, latex = [], []
    R = {}
    for name in args.random:
        p = os.path.join(args.out, name, "replicates.csv")
        if os.path.exists(p):
            R[name.replace("_random", "")] = pd.read_csv(p)
    for name in args.trained:
        d = os.path.join(args.out, name)
        df = pd.read_csv(os.path.join(d, "replicates.csv"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        st = per_rep_stats(df)
        pt, bs = st.loc[0], st.loc[1:]
        reps = len(bs)
        lines.append(f"\n=== {name} ({meta['weights']}), {meta['source']}, {meta['n_blocks']} blocks x "
                     f"{meta['block_images']} images, {reps} replicates ===")
        lines.append(f"  gate-passing dense conv layers: {int(pt.n_layers)}")
        tex = [("model", name)]

        def add(label, key, integer=False, nd=3):
            if key in st and not np.isnan(pt[key]):
                s = _fmt(pt[key], _ci(bs[key]), nd=nd, integer=integer)
                lines.append(f"  {label:58s} {s}")
                tex.append((label, key, s))

        add("pooled lower (count of pooled-gate layers)", "pool_lower", integer=True)
        lines.append(f"  {'pooled-gate layers':58s} {int(pt.get('pool_n', 0))}")
        add("pooling drop /C (median over layers)", "pool_drop_c")
        add("pooling drop /r_max", "pool_drop_r")
        add("jitter /C", "pool_jit_c")
        add("jitter /r_max", "pool_jit_r")
        add("drop-to-jitter ratio /C", "pool_ratio_c", nd=2)
        add("drop-to-jitter ratio /r_max", "pool_ratio_r", nd=2)
        add("max k*(.999)/r_max", "rmax_max")
        add("bounded - unbounded median k*(.999)/C", "rmax_diff_c")
        add("bounded - unbounded median k*(.999)/r_max", "rmax_diff_r")
        if "rmax_p_c" in st:
            lines.append(f"  {'Mann-Whitney p < 0.01 under C / under r_max (fraction of reps)':58s} "
                         f"{(bs.rmax_p_c < 0.01).mean():.2f} / {(bs.rmax_p_r < 0.01).mean():.2f}"
                         f"   (point: {pt.rmax_p_c:.1e} / {pt.rmax_p_r:.1e})")
        add("block output above conv3 bound (count)", "block_over", integer=True)
        if "block_n" in st:
            lines.append(f"  {'bottleneck blocks compared':58s} {int(pt.block_n)}")
        add("median block/conv3 ratio, k*(.999)/C", "block_ratio", nd=2)
        add("level: median k*(.95)/C", "level_c")
        add("level: median k*(.95)/r_max", "level_r")
        add("rho(depth, k*(.95)/C)", "rho_depth", nd=2)

        hw = per_layer_halfwidths(df, (PRIMARY, "k_star_rmax_0.95", K999, "k_star_rmax_0.999", "pooled_" + PRIMARY))
        for c, (med, mx) in hw.items():
            lines.append(f"  per-layer 95% CI half-width, {c:28s} median {med:.4f}  max {mx:.4f}")
            tex.append((f"half-width {c}", c, f"{med:.3f} / {mx:.3f}"))

        base = name
        if base in R:
            tr = trained_vs_random(df, R[base], (PRIMARY, "k_star_rmax_0.95", K999, "k_star_rmax_0.999"))
            lines.append(f"  trained vs random ({base}_random, same block draws), {int(tr.loc[0, 'n'])} common layers:")
            for c in (PRIMARY, "k_star_rmax_0.95", K999, "k_star_rmax_0.999"):
                s = _fmt(tr.loc[0, f"tr_{c}"], _ci(tr.loc[1:, f"tr_{c}"]), nd=2)
                s2 = _fmt(tr.loc[0, f"tr_meddiff_{c}"], _ci(tr.loc[1:, f"tr_meddiff_{c}"]))
                lines.append(f"    rho T-R {c:22s} {s}     median T-R difference {s2}")
                tex.append((f"rho T-R {c}", c, s))
                tex.append((f"T-R diff {c}", c, s2))
            rs = per_rep_stats(R[base])
            lines.append(f"    random level k*(.95)/C {_fmt(rs.loc[0, 'level_c'], _ci(rs.loc[1:, 'level_c']))}"
                         f"   random rho(depth) {_fmt(rs.loc[0, 'rho_depth'], _ci(rs.loc[1:, 'rho_depth']), nd=2)}")
        ap = os.path.join(d, "align.csv")
        if os.path.exists(ap):
            al = align_stats(pd.read_csv(ap))
            s = _fmt(al.loc[0, "median"], _ci(al.loc[1:, "median"]), nd=2)
            lines.append(f"  rho_align median over {int(al.loc[0, 'n'])} dense convs {s}   "
                         f"neg/pos {int(al.loc[0, 'neg'])}/{int(al.loc[0, 'pos'])}  "
                         f"(pos count CI {_fmt(al.loc[0, 'pos'], _ci(al.loc[1:, 'pos']), integer=True)})")
            tex.append(("rho_align median", "align", s))
            if base in R:
                arp = os.path.join(args.out, base + "_random", "align.csv")
                if os.path.exists(arp):
                    ar = align_stats(pd.read_csv(arp))
                    lines.append(f"    random rho_align median {_fmt(ar.loc[0, 'median'], _ci(ar.loc[1:, 'median']), nd=2)}")
        latex.append(tex)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    if args.latex:
        write_latex(latex, args.latex)


LATEX_ROWS = [
    ("pooled lower (count of pooled-gate layers)", r"Pooled lower than positional, count of gated layers"),
    ("pooling drop /r_max", r"Pooling drop, median over layers, $k^*/r_{\max}$"),
    ("drop-to-jitter ratio /r_max", r"Drop-to-jitter ratio, $k^*/r_{\max}$"),
    ("max k*(.999)/r_max", r"$\max_\ell\, k^*(0.999)/r_{\max}$"),
    ("bounded - unbounded median k*(.999)/C", r"Bounded $-$ unbounded median, $k^*(0.999)/C$"),
    ("bounded - unbounded median k*(.999)/r_max", r"Bounded $-$ unbounded median, $k^*(0.999)/r_{\max}$"),
    ("block output above conv3 bound (count)", r"Block output above the conv3 bound, count"),
    ("median block/conv3 ratio, k*(.999)/C", r"Block/conv3 ratio, median"),
    ("level: median k*(.95)/r_max", r"Level, median $k^*(0.95)/r_{\max}$"),
    ("rho T-R k_star_rmax_0.95", r"$\rho$(trained, random), $k^*(0.95)/r_{\max}$"),
    ("T-R diff k_star_rmax_0.95", r"Trained $-$ random, median over layers"),
    ("rho_align median", r"$\rho_{\mathrm{align}}$, median over dense convs"),
    ("half-width k_star_ratio_0.95", r"Per-layer CI half-width $k^*(0.95)/C$, median / max"),
    ("half-width k_star_rmax_0.999", r"Per-layer CI half-width $k^*(0.999)/r_{\max}$, median / max"),
]


def write_latex(blocks: list, path: str) -> None:
    names = [b[0][1] for b in blocks]
    look = [{lab: val for lab, _k, val in b[1:]} for b in blocks]
    col = " ".join("c" for _ in names)
    L = [r"\begin{tabular}{l " + col + "}", r"\toprule",
         "Statistic & " + " & ".join(n.replace("_", r"\_") for n in names) + r" \\", r"\midrule"]
    for key, label in LATEX_ROWS:
        vals = [lk.get(key, "--") for lk in look]
        if all(v == "--" for v in vals):
            continue
        L.append(label + " & " + " & ".join(v.replace("[", "$[$").replace("]", "$]$") for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--block-images", type=int, default=200)
    c.add_argument("--batch-size", type=int, default=50)
    c.add_argument("--positions", type=int, default=16)
    c.add_argument("--patch-positions", type=int, default=48)
    c.add_argument("--image-size", type=int, default=224)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--threads", type=int, default=6)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--min-n-over-C", type=float, default=50.0)
    c.add_argument("--force", action="store_true")
    b = sub.add_parser("bootstrap")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--reps", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--threads", type=int, default=6)
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("summarise")
    s.add_argument("--out", required=True)
    s.add_argument("--trained", nargs="+", required=True)
    s.add_argument("--random", nargs="*", default=[])
    s.add_argument("--latex", default=None)
    args = p.parse_args(argv)
    {"collect": collect, "bootstrap": bootstrap, "summarise": summarise}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
