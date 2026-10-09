"""Plan 9.47 (P9.110-P9.112): the four kinds of "dimension" on the same layers -- linear tau-width against the
manifold intrinsic dimension (TwoNN, Facco et al. 2017), with the effect of dependent position samples made explicit.

For every gate-passing dense convolution output of a checkpoint, four sample sets are drawn from the same images
and loader as the main measurement:

    S1  6,400 images x 1 random position          (independent across images)
    S3  the first 1,600 images of S1              (n control: same images as S2, one position each)
    S2  those 1,600 images x 16 random positions  (S3's position is the first of the 16; dependent within an image)
    P   6,400 global-average vectors              (one per image)

TwoNN: mu = r2 / r1 over the points, exact duplicate rows removed first, the largest 10 % of mu dropped, and the slope
through the origin of -log(1 - F(mu)) against log mu.  The linear scale (k*(0.9), k*(0.95), k*(0.999), r_max) is
read from the stored *_layers.csv of the same checkpoint (n = 102,400, 16 positions).

    cd code && python scripts/measure_twonn.py --results ../results --data ../data/imagenet_val_6400
    cd code && python scripts/measure_twonn.py --results ../results --judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINED = ("timm:resnet18.tv_in1k", "timm:resnet50.tv_in1k", "timm:vgg16_bn.tv_in1k")
RANDOM = ("timm:resnet18.tv_in1k_random", "timm:vgg16_bn.tv_in1k_random")
N_DEP_IMAGES = 1600
POSITIONS = 16
TRIM = 0.10
JITTER_REPS = 5


def stem(name: str) -> str:
    return name.replace("timm:", "timm_")


def stored_layers_csv(results: str, name: str) -> str:
    """Where the main measurement of this checkpoint lives (plan 9.47)."""
    s = stem(name)
    if name.endswith("_random"):
        return os.path.join(results, "local6400", "archs", "seed1", f"{s}_layers.csv")
    cand = os.path.join(results, "local6400", "archs", "trained", f"{s}_layers.csv")
    if os.path.exists(cand):
        return cand
    return os.path.join(results, f"{s}_layers.csv")


# ----------------------------------------------------------------------------------------------------- estimators

def kstar(eig: np.ndarray, tau: float) -> int:
    eig = np.clip(np.asarray(eig, dtype=np.float64), 0, None)
    if eig.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eig) / eig.sum(), tau) + 1)


def two_nearest(X: np.ndarray, device: str = "cpu", chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the first and second nearest *other* point (by index), exact, chunked over queries."""
    Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)
    n = Xt.shape[0]
    sq = (Xt * Xt).sum(1)
    r1 = torch.empty(n, device=device)
    r2 = torch.empty(n, device=device)
    ar = torch.arange(n, device=device)
    for a in range(0, n, chunk):
        q = Xt[a:a + chunk]
        d2 = sq[a:a + chunk, None] + sq[None, :] - 2.0 * (q @ Xt.T)
        d2.clamp_(min=0.0)
        d2[torch.arange(q.shape[0], device=device), ar[a:a + chunk]] = float("inf")   # not itself
        v, _ = torch.topk(d2, 2, dim=1, largest=False)
        r1[a:a + chunk] = v[:, 0]
        r2[a:a + chunk] = v[:, 1]
    return r1.sqrt().cpu().numpy().astype(np.float64), r2.sqrt().cpu().numpy().astype(np.float64)


def twonn_from_ratios(r1: np.ndarray, r2: np.ndarray, trim: float = TRIM) -> tuple[float, int]:
    """Facco et al. 2017: slope through the origin of -log(1 - F(mu)) vs log mu, F the empirical CDF of mu = r2/r1,
    exact duplicates (r1 = 0) dropped, the top `trim` fraction of mu dropped.  Returns (id, points used)."""
    keep = r1 > 0
    mu = np.sort(r2[keep] / r1[keep])
    n = mu.size
    if n < 10:
        return float("nan"), int(n)
    F = np.arange(1, n + 1) / n
    m = int(np.floor(n * (1.0 - trim)))
    x = np.log(mu[:m])
    y = -np.log(1.0 - F[:m])
    d = float((x * y).sum() / (x * x).sum())
    return d, int(m)


def twonn(X: np.ndarray, device: str = "cpu", trim: float = TRIM) -> tuple[float, int]:
    """Exact duplicate rows are removed first (Facco et al. 2017): a duplicated pair sits at one distance from a third
    point, gives it mu = 1 and would drag the slope to zero."""
    X = np.unique(np.asarray(X, dtype=np.float32), axis=0)
    r1, r2 = two_nearest(X, device)
    return twonn_from_ratios(r1, r2, trim)


def twonn_jitter(X: np.ndarray, rng: np.random.Generator, device: str = "cpu", reps: int = JITTER_REPS) -> float:
    """Standard deviation of the estimate over `reps` half-size subsamples (the curve's own jitter, rule 9)."""
    n = X.shape[0]
    vals = []
    for _ in range(reps):
        idx = rng.choice(n, n // 2, replace=False)
        vals.append(twonn(X[idx], device, )[0])
    return float(np.nanstd(vals))


# ----------------------------------------------------------------------------------------------------- collection

class SampleCollector:
    """Forward hooks on every dense nn.Conv2d output; keeps S1 / S2 / P as float16 on the CPU."""

    def __init__(self, model: nn.Module, wanted: set[str], seed: int, n_dep_images: int = N_DEP_IMAGES,
                 positions: int = POSITIONS):
        self.wanted = wanted
        self.n_dep = n_dep_images
        self.p = positions
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.seen = 0                      # images consumed so far (set by the caller per batch)
        self.s1: dict[str, list] = {}
        self.s2: dict[str, list] = {}
        self.pool: dict[str, list] = {}
        self.meta: dict[str, dict] = {}
        self.calls: dict[str, int] = {}
        self.handles = [model.register_forward_pre_hook(lambda _m, _i: self.calls.clear())]
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.groups == 1:
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_m, _i, out):
            idx = self.calls.get(name, 0)
            self.calls[name] = idx + 1
            key = f"{name}#{idx}"
            if key not in self.wanted or out.dim() != 4:
                return
            N, C, H, W = out.shape
            hw = H * W
            flat = out.detach().permute(0, 2, 3, 1).reshape(N, hw, C)
            p = min(self.p, hw)
            pos = torch.randint(0, hw, (N, p), generator=self.gen).to(flat.device)
            g = torch.gather(flat, 1, pos.unsqueeze(-1).expand(N, p, C))          # (N, p, C)
            self.s1.setdefault(key, []).append(g[:, 0].to(torch.float16).cpu())
            n_dep = max(0, min(N, self.n_dep - self.seen))
            if n_dep > 0:
                self.s2.setdefault(key, []).append(g[:n_dep].reshape(n_dep * p, C).to(torch.float16).cpu())
            self.pool.setdefault(key, []).append(flat.mean(1).to(torch.float16).cpu())
            self.meta.setdefault(key, {"C": C, "H": H, "W": W, "p": p})
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def arrays(self, key: str) -> dict[str, np.ndarray]:
        s1 = torch.cat(self.s1[key]).float().numpy()
        s2 = torch.cat(self.s2[key]).float().numpy()
        pl = torch.cat(self.pool[key]).float().numpy()
        p = self.meta[key]["p"]
        s3 = s2.reshape(-1, p, s2.shape[1])[:, 0]                                 # first position of each image
        return {"S1": s1, "S2": s2, "S3": s3, "P": pl}


def gate_rows(layers_csv: str) -> pd.DataFrame:
    df = pd.read_csv(layers_csv)
    df = df[(df["kind"] == "conv") & (df["groups"] == 1) & (~df["is_depthwise"].astype(bool)) & df["n_over_C_ok"].astype(bool)]
    return df.copy()


@torch.no_grad()
def measure(name: str, results: str, data: str, device: str, seed: int = 1, batch_size: int = 32,
            limit: int | None = None) -> pd.DataFrame:
    from layerspec.data import build_loader
    from layerspec.models import build

    t0 = time.time()
    rows = gate_rows(stored_layers_csv(results, name))
    wanted = {f"{r['layer']}#{int(r['call_index'])}" for r in rows.to_dict("records")}
    model, tag = build(name, pretrained=not name.endswith("_random"), seed=seed)
    model = model.to(device).eval()
    loader, src = build_loader(data, batch_size=batch_size, num_workers=4, image_size=224, seed=0, preprocess="crop",
                               limit=limit)
    col = SampleCollector(model, wanted, seed=0)
    try:
        for i, batch in enumerate(loader):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(images.to(device))
            col.seen += images.shape[0]
            if i % 40 == 0:
                print(f"  {name}: batch {i}", flush=True)
    finally:
        col.remove()
    print(f"  {name}: forward done in {time.time() - t0:.0f} s; {len(col.s1)} layers", flush=True)
    rng = np.random.default_rng(0)
    out = []
    for r in rows.to_dict("records"):
        key = f"{r['layer']}#{int(r['call_index'])}"
        if key not in col.s1:
            continue
        A = col.arrays(key)
        rec = {"model": name, "weights": tag, "source": src, "layer": r["layer"], "depth_index": int(r["depth_index"]),
               "C": int(r["C"]), "r_max": float(r["r_max"]), "n_over_C_ok": True,
               "k90": int(r["k_star_0.9"]), "k95": int(r["k_star_0.95"]),
               "k999": int(r["k_star_0.999"]), "k95_rmax": float(r["k_star_rmax_0.95"])}
        for s in ("S1", "S2", "S3", "P"):
            d, m = twonn(A[s], device)
            rec[f"id_{s}"] = d
            rec[f"n_{s}"] = int(A[s].shape[0])
            rec[f"used_{s}"] = m
            rec[f"id_{s}_sd"] = twonn_jitter(A[s], rng, device)
        if rec["C"] <= 128:                                                        # sanity: n/C >= 50 on S1 alone
            X = A["S1"].astype(np.float64)
            X = X - X.mean(0)
            rec["k95_S1_check"] = kstar(np.linalg.eigvalsh(X.T @ X / (X.shape[0] - 1))[::-1], 0.95)
        else:
            rec["k95_S1_check"] = np.nan
        out.append(rec)
        print(f"    {key:28s} C={rec['C']:5d} k95={rec['k95']:4d}  ID S1={rec['id_S1']:.1f} S2={rec['id_S2']:.1f} "
              f"S3={rec['id_S3']:.1f} P={rec['id_P']:.1f}", flush=True)
    df = pd.DataFrame(out)
    print(f"  {name}: done in {time.time() - t0:.0f} s", flush=True)
    return df


# ----------------------------------------------------------------------------------------------------- judgement

def spearman(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 3:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def ordering_class(rho: float) -> str:
    if np.isnan(rho):
        return "n/a"
    return "same" if rho >= 0.8 else ("partial" if rho >= 0.5 else "different")


def judge_frame(df: pd.DataFrame) -> dict:
    """P9.110-P9.112 and the exploratory lines, from the per-layer table(s)."""
    tr = df[~df["model"].str.endswith("_random")]
    res: dict = {}
    # P9.110 ladder: ID(S1) < k*(0.90) in >= 95 % of gate layers (trained checkpoints pooled)
    ok = (tr["id_S1"] < tr["k90"])
    res["P9.110"] = {"pass": int(ok.sum()), "total": int(len(tr)), "frac": float(ok.mean()) if len(tr) else float("nan"),
                     "holds": bool(len(tr) and ok.mean() >= 0.95),
                     "violations": tr.loc[~ok, ["model", "layer", "id_S1", "k90"]].to_dict("records")}
    # P9.111 dependence: ID(S2) < ID(S3) in >= 80 % of gate layers (trained)
    dep = (tr["id_S2"] < tr["id_S3"])
    res["P9.111"] = {"pass": int(dep.sum()), "total": int(len(tr)), "frac": float(dep.mean()) if len(tr) else float("nan"),
                     "holds": bool(len(tr) and dep.mean() >= 0.80),
                     "median_ratio_S2_over_S3": float((tr["id_S2"] / tr["id_S3"]).median()) if len(tr) else float("nan"),
                     "S2_lt_S1_frac": float((tr["id_S2"] < tr["id_S1"]).mean()) if len(tr) else float("nan")}
    # P9.112 ordering: rho(ID(S1), k95/r_max) < 0.5 in 3/3 trained checkpoints
    per = {}
    for m, g in tr.groupby("model"):
        rho = spearman(g["id_S1"], g["k95_rmax"])
        per[m] = {"rho_rmax": rho, "class": ordering_class(rho), "rho_raw": spearman(g["id_S1"], g["k95"]),
                  "n_layers": int(len(g))}
    res["P9.112"] = {"per_model": per,
                     "different": sum(1 for v in per.values() if v["class"] == "different"),
                     "n_models": len(per), "holds": bool(per) and all(v["class"] == "different" for v in per.values())}
    # exploratory
    ex = {}
    for m, g in df.groupby("model"):
        ex[m] = {"id_S1_median": float(g["id_S1"].median()), "id_P_median": float(g["id_P"].median()),
                 "id_S1_over_rmax_median": float((g["id_S1"] / g["r_max"]).median()),
                 "rho_P_vs_S1": spearman(g["id_P"], g["id_S1"]),
                 "k95_check_max_abs_diff": float((g["k95_S1_check"] - g["k95"]).abs().max()) if g["k95_S1_check"].notna().any() else float("nan")}
    res["exploratory"] = ex
    return res


def format_judgement(res: dict) -> str:
    L = []
    a = res["P9.110"]
    L.append(f"P9.110 ladder ID(S1) < k*(0.90): {a['pass']}/{a['total']} ({a['frac']:.3f}) -> {'HOLDS' if a['holds'] else 'FAILS'} (>= 0.95)")
    for v in a["violations"]:
        L.append(f"    violation {v['model']} {v['layer']}: ID {v['id_S1']:.1f} vs k90 {v['k90']}")
    b = res["P9.111"]
    L.append(f"P9.111 ID(S2) < ID(S3): {b['pass']}/{b['total']} ({b['frac']:.3f}) -> {'HOLDS' if b['holds'] else 'FAILS'} (>= 0.80); "
             f"median ID(S2)/ID(S3) {b['median_ratio_S2_over_S3']:.2f}; ID(S2) < ID(S1) frac {b['S2_lt_S1_frac']:.2f}")
    c = res["P9.112"]
    L.append(f"P9.112 rho(ID(S1), k95/r_max) < 0.5 in {c['different']}/{c['n_models']} -> {'HOLDS' if c['holds'] else 'FAILS'}")
    for m, v in c["per_model"].items():
        L.append(f"    {m}: rho_rmax {v['rho_rmax']:+.2f} ({v['class']}), rho_raw {v['rho_raw']:+.2f}, layers {v['n_layers']}")
    L.append("exploratory:")
    for m, v in res["exploratory"].items():
        L.append(f"    {m}: ID(S1) med {v['id_S1_median']:.1f}, ID(P) med {v['id_P_median']:.1f}, ID/r_max med {v['id_S1_over_rmax_median']:.3f}, "
                 f"rho(P,S1) {v['rho_P_vs_S1']:+.2f}, k95 sanity max|diff| {v['k95_check_max_abs_diff']:.0f}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--data", default="../data/imagenet_val_6400")
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="*", default=list(TRAINED) + list(RANDOM))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default <results>/twonn)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: number of images")
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(args.results, "twonn")
    os.makedirs(out_dir, exist_ok=True)
    if args.judge:
        frames = [pd.read_csv(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir)) if f.endswith("_twonn.csv")]
        df = pd.concat(frames, ignore_index=True)
        res = judge_frame(df)
        txt = format_judgement(res)
        print(txt)
        with open(os.path.join(out_dir, "judgement.txt"), "w") as fh:
            fh.write(txt + "\n")
        with open(os.path.join(out_dir, "judgement.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        return
    from layerspec.api import select_device
    device = select_device(args.device)
    for name in args.models:
        path = os.path.join(out_dir, f"{stem(name)}_twonn.csv")
        if os.path.exists(path):
            print(f"skip {name}: {path} exists")
            continue
        df = measure(name, args.results, args.data, device, limit=args.limit)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
