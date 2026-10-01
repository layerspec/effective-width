"""B7, second stage (plan 9.27 A/B; theory-chain note B7.2'): the memory-window gradient Gram against the final output
covariance, on runs that saved checkpoints along the trajectory (the ruler-width arms of A18, the weight-decay sweep).

For every saved checkpoint E of a run, on 6,400 CIFAR-10 training images x 16 positions (BN in training mode):
    Sigma_E   channel covariance of each convolution's output      Gamma_E   second moment of the loss gradient w.r.t. it
(the accumulators of measure_gradgram.gradient_grams).  The optimiser's impulse response (Lemma B7.1: W_T = a_T W_0 - sum_s c_s g_s,
c_s >= 0, computed exactly for the run's OneCycle schedule and its weight decay) assigns every step to the nearest saved
checkpoint, w_E = sum of c_s over those steps, and
    Gamma_win = sum_E w_E Gamma_E         Gamma_end = Gamma_T         Gamma_avg = equal-weight mean over the saved E >= 10.
Per layer the tau-widths k*(0.95), k*(0.999) of Sigma_T (final), Gamma_win, Gamma_end, Gamma_avg are written to <out>/<run>_layers.csv,
the window weights and the half-mass epoch to <out>/<run>_weights.json; the per-checkpoint Grams are cached as .npz (resumable).

    python scripts/measure_gradwindow.py --run ../results/a18_pt/vgg16_bn_ruler_s0 --lam 5e-4 --out ../results/gradwindow
    python scripts/measure_gradwindow.py --judge ruler --out ../results/gradwindow        # P-b7.1'
    python scripts/measure_gradwindow.py --judge sweep --out ../results/gradwindow        # P-b7.4a-d
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_gradgram import TAUS, gradient_grams, k_star          # noqa: E402
from scripts.measure_stationarity import conv_layers, labelled_loader      # noqa: E402

STEPS_PER_EPOCH = 391            # CIFAR-10, 50,000 images, batch 128 (trajectory_cifar.py)
WINDOW_EPOCHS = (40, 50, 60, 70, 80, 90, 100)
WIDE = 100                       # layers judged in P-b7.1'(a)(c): at least this many channels
LAMS = (5e-5, 5e-4, 5e-3)


def impulse_response(lrs, moms, lam: float, nesterov: bool = True):
    """(a_T, c_s) of torch.optim.SGD with weight decay: d = g + lam*w; buf = mu*buf + d; w -= lr*(d + mu*buf) [Nesterov]."""
    T = len(lrs)
    c_w = np.zeros(T); c_b = np.zeros(T); a_w, a_b = 1.0, 0.0
    for t in range(T):
        mu, lr = moms[t], lrs[t]
        d_w = lam * c_w.copy(); d_w[t] += 1.0
        d_a = lam * a_w
        if t == 0:
            c_b = d_w.copy(); a_b = d_a
        else:
            c_b = mu * c_b + d_w; a_b = mu * a_b + d_a
        step_w = d_w + mu * c_b if nesterov else c_b
        step_a = d_a + mu * a_b if nesterov else a_b
        c_w = c_w - lr * step_w; a_w = a_w - lr * step_a
    return a_w, -c_w


def onecycle_schedule(epochs: int = 100, steps: int = STEPS_PER_EPOCH, max_lr: float = 0.1):
    """Per-step (lr, momentum) of trajectory_cifar.py's OneCycleLR (opt.step() then sched.step() every batch)."""
    dummy = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([dummy], lr=max_lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps)
    lrs, moms = [], []
    for _ in range(epochs * steps):
        lrs.append(opt.param_groups[0]["lr"]); moms.append(opt.param_groups[0]["momentum"])
        opt.step(); sched.step()
    return np.asarray(lrs), np.asarray(moms)


def window_weights(saved_epochs, lam: float, epochs: int = 100, steps: int = STEPS_PER_EPOCH) -> tuple[dict, dict]:
    """w_E (sum to one) over the saved checkpoint epochs, and the mass quantiles of c_s in epochs."""
    lrs, moms = onecycle_schedule(epochs, steps)
    a, c = impulse_response(lrs, moms, lam)
    step_epoch = np.arange(len(c)) // steps + 1                     # the gradient at step s is taken during epoch e
    saved = np.asarray(sorted(int(e) for e in saved_epochs))
    d = np.abs(step_epoch[:, None] - saved[None, :]).astype(float)
    d -= 1e-6 * np.arange(len(saved))[None, :]                      # ties go to the later checkpoint
    nearest = saved[np.argmin(d, axis=1)]
    w = {int(E): float(c[nearest == E].sum()) for E in saved}
    tot = sum(w.values())
    w = {E: v / tot for E, v in w.items()}
    cum = np.cumsum(c) / c.sum()
    q = {f"q{int(100 * t)}": int(step_epoch[min(int(np.searchsorted(cum, t)), len(c) - 1)]) for t in (0.1, 0.5, 0.9)}
    return w, dict(a_T=float(a), **q)


def build_from_checkpoint(path: str, num_classes: int = 10):
    from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    widths = [int(v.shape[0]) for k, v in sd.items() if k.startswith("features") and k.endswith("weight") and v.dim() == 4]
    m = build_vgg16_bn_cifar_widths(widths, num_classes)
    m.load_state_dict(sd)
    return m, widths


def _epoch_of(path: str) -> int:
    return int(re.search(r"epoch_(\d+)\.pt$", path).group(1))


def measure_run(run: str, lam: float, out: str, data: str, device: str, epochs=WINDOW_EPOCHS, n_images: int = 6400,
                positions_seed: int = 0) -> pd.DataFrame:
    name = os.path.basename(os.path.normpath(run))
    os.makedirs(out, exist_ok=True)
    ckpts = {_epoch_of(p): p for p in glob.glob(os.path.join(run, "epoch_*.pt")) if _epoch_of(p) in epochs}
    if not ckpts or max(ckpts) != max(epochs):
        raise SystemExit(f"{run}: need checkpoints at epochs {epochs}, found {sorted(ckpts)}")
    loader, K = labelled_loader(data, "cifar10", None, None, n_images=n_images)
    grams = {}
    for E in sorted(ckpts):
        cache = os.path.join(out, f"{name}_epoch_{E:03d}.npz")
        if os.path.exists(cache):
            z = np.load(cache); grams[E] = {k: z[k] for k in z.files}; continue
        t0 = time.time()
        model, _ = build_from_checkpoint(ckpts[E], K)
        model.to(device)
        sig, gam = gradient_grams(model, loader, device, seed=positions_seed)
        g = {}
        for n in sig:
            g[n + "::out"] = sig[n].numpy().astype(np.float32); g[n + "::grad"] = gam[n].numpy().astype(np.float32)
        np.savez_compressed(cache + ".tmp.npz", **g); os.replace(cache + ".tmp.npz", cache)
        grams[E] = g
        print(f"  {name} epoch {E}: {time.time() - t0:.0f} s", flush=True)
    model, widths = build_from_checkpoint(ckpts[max(ckpts)], K)
    layers = [n for n, _ in conv_layers(model)]
    T = max(ckpts)
    w, q = window_weights(sorted(ckpts), lam)
    avg_epochs = [E for E in sorted(ckpts) if E >= 10]
    rows = []
    for i, n in enumerate(layers):
        win = sum(w[E] * grams[E][n + "::grad"].astype(np.float64) for E in sorted(ckpts))
        end = grams[T][n + "::grad"].astype(np.float64)
        avg = sum(grams[E][n + "::grad"].astype(np.float64) for E in avg_epochs) / len(avg_epochs)
        sig = grams[T][n + "::out"].astype(np.float64)
        row = dict(index=i, layer=n, C=widths[i])
        for tag, M in (("out", sig), ("win", win), ("end", end), ("avg", avg)):
            e = np.clip(np.linalg.eigvalsh(M)[::-1], 0, None)
            for tau in TAUS:
                row[f"k{tau}_{tag}"] = k_star(e, tau)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{name}_layers.csv"), index=False)
    acc = None
    f = os.path.join(run, f"epoch_{T:03d}_layers.csv")
    if os.path.exists(f):
        acc = float(pd.read_csv(f).test_acc.iloc[0])
    with open(os.path.join(out, f"{name}_weights.json"), "w") as fh:
        json.dump(dict(run=name, lam=lam, weights={str(k): v for k, v in w.items()}, **q, final_epoch=T, test_acc=acc,
                       widths=widths, n_images=n_images), fh, indent=1)
    late = df.iloc[-3:]
    print(f"=== {name} (lam {lam:g}): window mass q10/q50/q90 at epochs {q['q10']}/{q['q50']}/{q['q90']}; late k0.999 out {late['k0.999_out'].tolist()} "
          f"win {late['k0.999_win'].tolist()} end {late['k0.999_end'].tolist()} acc {acc} ===", flush=True)
    return df


def _checks(df: pd.DataFrame) -> dict:
    """P-b7.1'(a)(b)(c) and the descriptive distances of one run."""
    from scipy.stats import spearmanr
    wide = df[df.C >= WIDE]
    out = {}
    out["a_rho"] = float(spearmanr(wide["k0.95_win"], wide["k0.999_out"]).correlation) if len(wide) >= 4 else float("nan")
    out["a"] = bool(out["a_rho"] >= 0.7)
    out["b_viol"] = int((df["k0.999_out"] > df["k0.999_win"] + 1).sum()); out["b"] = out["b_viol"] == 0
    out["c_ratio"] = float(np.median(wide["k0.999_out"] / wide["k0.999_win"])); out["c"] = bool(out["c_ratio"] >= 0.5)
    for tag in ("end", "avg"):
        out[f"rho_{tag}"] = float(spearmanr(wide[f"k0.95_{tag}"], wide["k0.999_out"]).correlation) if len(wide) >= 4 else float("nan")
        out[f"d_{tag}"] = float(np.median(np.abs(np.log(wide["k0.999_out"]) - np.log(wide[f"k0.999_{tag}"]))))
    out["d_win"] = float(np.median(np.abs(np.log(wide["k0.999_out"]) - np.log(wide["k0.999_win"]))))
    out["late_level"] = float(np.median(df.iloc[-3:]["k0.999_out"] / df.iloc[-3:].C))
    return out


def judge_ruler(out: str) -> dict:
    files = sorted(glob.glob(os.path.join(out, "vgg16_bn_ruler_s[0-9]_layers.csv")))
    print("==== P-b7.1' (plan 9.27 A): the memory-window Gram against the final output covariance, A18 ruler-width arms ====")
    if not files:
        print("  nothing measured"); return {}
    R = {}
    for f in files:
        name = os.path.basename(f).replace("_layers.csv", ""); df = pd.read_csv(f); ck = _checks(df); R[name] = ck
        q = json.load(open(f.replace("_layers.csv", "_weights.json")))
        print(f"  {name}: (a) rho(k0.95 win, k0.999 out | C >= {WIDE}) {ck['a_rho']:+.2f}  (b) violations {ck['b_viol']}/{len(df)}  "
              f"(c) ratio median {ck['c_ratio']:.2f}  | rho with end {ck['rho_end']:+.2f} avg {ck['rho_avg']:+.2f}; "
              f"|dlog| win {ck['d_win']:.2f} end {ck['d_end']:.2f} avg {ck['d_avg']:.2f}; window q10/q50/q90 {q['q10']}/{q['q50']}/{q['q90']}, "
              f"mass at E >= 60: {sum(v for k, v in q['weights'].items() if int(k) >= 60):.2f}")
    V = {}
    if len(R) >= 3:
        for key, text in (("a", f"Spearman(k0.95 Gamma_win, k0.999 Sigma_out) >= +0.7 over layers with C >= {WIDE}"),
                          ("b", "k0.999(Sigma_out) <= k0.999(Gamma_win) + 1 in every layer"), ("c", "median ratio >= 0.5 (tightness)")):
            V[f"P-b7.1'({key})"] = all(r[key] for r in R.values())
            print(f"  P-b7.1'({key}) ({text}, 3/3) -> {'holds' if V[f'P-b7.1({key})'.replace('P-b7.1(', chr(80)+'-b7.1'+chr(39)+'(')] else 'fails'}"
                  if False else f"  P-b7.1'({key}) ({text}, 3/3) -> {'holds' if V[f'P-b7.1' + chr(39) + f'({key})'] else 'fails'}")
        if all(V.values()):
            print("  Registered consequence: B7.2' enters chain link 3 as a measured inequality (proposition + table)")
        elif not V["P-b7.1'(b)"]:
            print("  Registered consequence: hypothesis B7.2 fails (the lemma stands); measure cond(M) next")
        elif not V["P-b7.1'(c)"]:
            print("  Registered consequence: the bound is loose; B7 is written as an ordering (P-b7.2), not a bound")
        else:
            print("  Registered consequence: (a) fails; the ordering across layers is not the window Gram's -- reported as measured")
    else:
        print("  fewer than three seeds -- not judged yet")
    return V


def judge_sweep(out: str) -> dict:
    files = sorted(glob.glob(os.path.join(out, "wd*_s[0-9]_layers.csv")))
    print("==== P-b7.4 (plan 9.27 B): the weight-decay sweep, full-width VGG-16 on CIFAR-10 ====")
    if not files:
        print("  nothing measured"); return {}
    R = {}
    for f in files:
        name = os.path.basename(f).replace("_layers.csv", ""); q = json.load(open(f.replace("_layers.csv", "_weights.json")))
        lam, seed = float(q["lam"]), int(name.rsplit("_s", 1)[1])
        R[(lam, seed)] = dict(_checks(pd.read_csv(f)), q10=q["q10"], q50=q["q50"], q90=q["q90"], acc=q["test_acc"])
        r = R[(lam, seed)]
        print(f"  lam {lam:g} seed {seed}: window q10/q50/q90 {r['q10']}/{r['q50']}/{r['q90']}  (a) {r['a_rho']:+.2f}  (b) viol {r['b_viol']}  "
              f"(c) {r['c_ratio']:.2f}  d_win {r['d_win']:.2f} d_end {r['d_end']:.2f} d_avg {r['d_avg']:.2f}  late level {r['late_level']:.3f}  acc {r['acc']}")
    lams = sorted({k[0] for k in R}); V = {}
    q50 = {l: np.median([r["q50"] for (ll, s), r in R.items() if ll == l]) for l in lams}
    print(f"  P-b7.4a (algebra, reported): half-mass epoch by lambda {', '.join(f'{l:g}: {q50[l]:.0f}' for l in lams)}")
    complete = all((l, s) in R for l in LAMS for s in (0, 1))
    if complete:
        V["P-b7.4b"] = all(r["a"] and r["b"] for r in R.values())
        print(f"  P-b7.4b (P-b7.1'(a)(b) in 6/6 runs) -> {'holds' if V['P-b7.4b'] else 'fails'}")
        hi = all(R[(5e-3, s)]["d_end"] < R[(5e-3, s)]["d_avg"] for s in (0, 1))
        lo = all(R[(5e-5, s)]["d_end"] > R[(5e-5, s)]["d_avg"] for s in (0, 1))
        V["P-b7.4c"] = hi and lo
        print(f"  P-b7.4c (5e-3: d_end < d_avg both seeds -> {hi}; 5e-5: d_end > d_avg both seeds -> {lo}) -> {'holds' if V['P-b7.4c'] else 'fails'}")
        accs = {k: r["acc"] for k, r in R.items()}
        comparable = all(a is not None for a in accs.values()) and all(abs(accs[(l, s)] - accs[(5e-4, s)]) <= 2.0 for l in LAMS for s in (0, 1))
        if comparable:
            V["P-b7.4d"] = all(R[(5e-5, s)]["late_level"] > R[(5e-4, s)]["late_level"] > R[(5e-3, s)]["late_level"] for s in (0, 1))
            print(f"  P-b7.4d (late level falls with lambda, both seeds) -> {'holds' if V['P-b7.4d'] else 'fails'}")
        else:
            print(f"  P-b7.4d not judged: accuracies not within 2.0 points of the 5e-4 arm ({accs})")
        if V["P-b7.4b"] and V["P-b7.4c"]:
            print("  Registered consequence: III-C gains 'the weight decay sets the window the width is read from' and the c_s figure")
        elif not V["P-b7.4c"]:
            print("  Registered consequence: the window-placement claim is withdrawn; B7 keeps 'the width follows the gradient Gram'")
    else:
        print("  not all six runs measured -- not judged yet")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=None, help="run directory with epoch_XXX.pt checkpoints")
    p.add_argument("--lam", type=float, default=5e-4, help="the run's weight decay (for the impulse response)")
    p.add_argument("--out", default="../results/gradwindow")
    p.add_argument("--data", default="../data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-images", type=int, default=6400)
    p.add_argument("--epochs", type=int, nargs="+", default=list(WINDOW_EPOCHS))
    p.add_argument("--judge", choices=("ruler", "sweep"), default=None)
    a = p.parse_args(argv)
    if a.judge == "ruler":
        judge_ruler(a.out); return 0
    if a.judge == "sweep":
        judge_sweep(a.out); return 0
    if a.run is None:
        p.error("--run or --judge")
    measure_run(a.run, a.lam, a.out, a.data, a.device, tuple(a.epochs), a.n_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
