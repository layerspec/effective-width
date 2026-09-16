"""P1, second half: build the NARROW network.  Keep the greedy column subset S_l of every
post-activation layer, fold the linear reconstruction of the dropped channels into the next
convolution (or the classifier), and evaluate the network that really has m_l channels.

  next conv:  W'[:, k] = W[:, k] + sum_{j in drop} B[k, j] W[:, j]   for k in S      (per kernel tap)
              b'       = b + sum_{j in drop} c_j * sum_taps W[:, j]
  classifier: same with the taps summed out (average pooling commutes with the reconstruction)

The fold is exact except (i) at the one-pixel zero-padding border, where the constant c_j is
absent from the padded taps, and (ii) across a max-pool, where reconstruction-then-pool (the hook
of cssp_check.py) and pool-then-reconstruction (the folded network) differ.  Both are reported
as the logit discrepancy between the hooked and the folded network.

    python scripts/fold_cssp.py --device mps --scheme a18_ruler_act_0.999 --out ../results/cssp
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cssp_check import (RELU_LAYERS, Patch, Stats, cssp_map, greedy_cssp,  # noqa: E402
                                kstar, load_model, loaders)
from scripts.reproduce_garg import CIFAR_STATS, evaluate                                        # noqa: E402
from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths                    # noqa: E402

CONV_OF = {r: f"features.{int(r.split('.')[1]) - 2}" for r in RELU_LAYERS}
BN_OF = {r: f"features.{int(r.split('.')[1]) - 1}" for r in RELU_LAYERS}


def fold(model, moments, subsets, widths, device, num_classes=10):
    """Return a VGG with the given widths whose weights are the folded ones."""
    narrow = build_vgg16_bn_cifar_widths(widths, num_classes=num_classes)
    src = dict(model.named_modules())
    dst = dict(narrow.named_modules())
    prev = None                                   # (S, B, c) of the previous post-activation layer
    with torch.no_grad():
        for r in RELU_LAYERS:
            conv_s, conv_d = src[CONV_OF[r]], dst[CONV_OF[r]]
            bn_s, bn_d = src[BN_OF[r]], dst[BN_OF[r]]
            W = conv_s.weight.detach().cpu().double()              # C_out x C_in x 3 x 3
            b = conv_s.bias.detach().cpu().double()
            if prev is not None:
                S, B, c = prev                                     # B: |S| x |drop|, c: |drop|
                drop = [j for j in range(W.shape[1]) if j not in set(S)]
                Wk = W[:, S] + torch.einsum("kj,ojhw->okhw", B, W[:, drop])
                b = b + torch.einsum("j,ojhw->o", c, W[:, drop])
                W = Wk
            S_out = subsets[r]
            conv_d.weight.copy_(W[S_out].float())
            conv_d.bias.copy_(b[S_out].float())
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(bn_d, name).copy_(getattr(bn_s, name)[S_out])
            bn_d.num_batches_tracked.copy_(bn_s.num_batches_tracked)
            mean, cov = moments[r]
            keep = torch.tensor(sorted(S_out))
            drop = torch.tensor([j for j in range(cov.shape[0]) if j not in set(S_out)])
            if len(drop):
                Sss = cov[keep][:, keep]
                B = torch.linalg.pinv(Sss, rtol=1e-8) @ cov[keep][:, drop]
                c = mean[drop] - B.T @ mean[keep]
            else:
                B = torch.zeros(len(keep), 0, dtype=torch.float64)
                c = torch.zeros(0, dtype=torch.float64)
            prev = (sorted(S_out), B, c)
        S, B, c = prev
        L_s, L_d = src["classifier"], dst["classifier"]
        W = L_s.weight.detach().cpu().double()                     # 10 x 512
        drop = [j for j in range(W.shape[1]) if j not in set(S)]
        L_d.weight.copy_((W[:, S] + W[:, drop] @ B.T).float())
        L_d.bias.copy_((L_s.bias.detach().cpu().double() + W[:, drop] @ c).float())
    return narrow.to(device).eval()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default="../results/round3_pt/a5_vgg16_bn_none_s0/vgg16_bn_cifar10_final.pt")
    p.add_argument("--widths-act", default="../results/a18/widths_act.json")
    p.add_argument("--data-root", default="../data")
    p.add_argument("--device", default="mps")
    p.add_argument("--n-calib", type=int, default=6400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schemes", nargs="+", default=["a18_ruler_act_0.999", "kstar_0.999", "kstar_0.99", "a18_ruler_act_0.95"])
    p.add_argument("--out", default="../results/cssp")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    model = load_model(a.checkpoint, a.device, CIFAR_STATS[a.dataset][2])
    calib, test = loaders(a.data_root, a.n_calib, a.batch_size, 0, a.seed, a.dataset)
    mods = dict(model.named_modules())
    stats = {n: Stats() for n in RELU_LAYERS}
    hs = [mods[n].register_forward_hook(lambda mod, inp, out, n=n: stats[n].update(out)) for n in RELU_LAYERS]
    with torch.no_grad():
        for x, _ in calib:
            model(x.to(a.device))
    for h in hs:
        h.remove()
    moments = {n: stats[n].finish() for n in RELU_LAYERS}
    spec = {n: np.clip(np.linalg.eigvalsh(moments[n][1].numpy())[::-1], 0, None) for n in RELU_LAYERS}
    pivots = {n: greedy_cssp(moments[n][1], moments[n][1].shape[0])[0] for n in RELU_LAYERS}
    with torch.no_grad():
        acc_full = evaluate(model, test, a.device)
    n_full = sum(p.numel() for p in model.parameters())
    print(f"full: {acc_full:.2f}%  {n_full / 1e6:.2f}M params", flush=True)

    widths_json = json.load(open(a.widths_act))["widths"]
    xb = next(iter(test))[0][:64].to(a.device)
    rows = []
    for scheme in a.schemes:
        if scheme == "a18_ruler_act_0.999":
            wd = dict(zip(RELU_LAYERS, widths_json["ruler"]))
        elif scheme == "a18_ruler_act_0.95":
            wd = dict(zip(RELU_LAYERS, widths_json["ruler95"]))
        elif scheme.startswith("kstar_"):
            tau = float(scheme.split("_")[1])
            wd = {n: min(kstar(spec[n], tau), spec[n].shape[0]) for n in RELU_LAYERS}
        else:
            raise ValueError(scheme)
        subsets = {n: sorted(pivots[n][:int(wd[n])]) for n in RELU_LAYERS}
        widths = [len(subsets[n]) for n in RELU_LAYERS]
        # hooked reference (reconstruction before pooling)
        maps = {n: cssp_map(*moments[n], subsets[n], a.device) for n in RELU_LAYERS if len(subsets[n]) < moments[n][1].shape[0]}
        pt = Patch(model, maps)
        with torch.no_grad():
            acc_hook = evaluate(model, test, a.device)
            logits_hook = model(xb).cpu()
        pt.remove()
        t0 = time.time()
        narrow = fold(model, moments, subsets, widths, a.device, CIFAR_STATS[a.dataset][2])
        with torch.no_grad():
            acc_fold = evaluate(narrow, test, a.device)
            logits_fold = narrow(xb).cpu()
            logits_full = model(xb).cpu()
        n_narrow = sum(p.numel() for p in narrow.parameters())
        d = (logits_fold - logits_hook).abs().max().item()
        d0 = (logits_fold - logits_full).abs().max().item()
        scale = logits_full.abs().max().item()
        rows.append({"scheme": scheme, "widths": " ".join(map(str, widths)), "params": n_narrow,
                     "param_frac": n_narrow / n_full, "acc_full": acc_full, "acc_hook": acc_hook,
                     "acc_fold": acc_fold, "max_logit_diff_fold_vs_hook": d, "max_logit_diff_fold_vs_full": d0,
                     "logit_scale": scale})
        print(f"{scheme:22s} widths {widths}\n    params {n_narrow / 1e6:.2f}M ({n_narrow / n_full:.3f})  "
              f"acc hook {acc_hook:.2f}  FOLDED {acc_fold:.2f}  (full {acc_full:.2f})  "
              f"max|logit diff| fold-hook {d:.3f}, fold-full {d0:.3f} on scale {scale:.1f}  [{time.time() - t0:.0f}s]",
              flush=True)
        torch.save({k: v.cpu() for k, v in narrow.state_dict().items()},
                   os.path.join(a.out, f"vgg16_bn_folded_{scheme}.pt"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fold_results.csv"), index=False)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
