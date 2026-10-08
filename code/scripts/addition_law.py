"""Plan 9.45 (P9.108): the exact sampling law at the residual addition.

In a timm Bottleneck at a fresh gamma = 1 initialisation in eval mode, y = x + W3 h exactly (conv3 is 1x1, BN3 is the
identity with fresh running statistics).  Given J = Cov([x; h]) the law of Sigma_y = [I W3] J [I W3]^T is a function of J
and the scale of W3 alone.  This script accumulates J for the blocks of the requested stages, then compares

    measured   k_tau(Sigma_y) read directly from the network
    rebuilt    the same from J with the network's own W3            (sanity: must agree)
    sampled    mean over fresh Gaussian W3 draws                     (P9.108a)
    no-cross   the same draws with the cross block C_xh set to zero  (P9.108b: the dependent cross term is necessary)

Usage: cd code && python scripts/addition_law.py [--judge] [--stages layer2 layer3 layer4]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAU = 0.95


def kstar(e, tau=TAU):
    e = np.clip(np.sort(np.asarray(e, dtype=float))[::-1], 0, None); c = np.cumsum(e) / e.sum(); return int(np.searchsorted(c, tau) + 1)


def sigma_y(J, C, W):
    """[I W] J [I W]^T for W of shape (C, Ch)."""
    T = np.concatenate([np.eye(C), W], 1)
    return T @ J @ T.T


def sampled_widths(J, C, std, draws, rng, cross=True, tau=TAU):
    Jc = J.copy()
    if not cross:
        Jc[:C, C:] = 0.0; Jc[C:, :C] = 0.0
    ks = []
    for _ in range(draws):
        W = rng.standard_normal((C, J.shape[0] - C)) * std
        ks.append(kstar(np.linalg.eigvalsh(sigma_y(Jc, C, W)), tau))
    return float(np.mean(ks)), float(np.std(ks))


def im2col(t, k, pad):
    """(n, c, h, w) -> (n, h*w, c*k*k) with zero padding and stride 1; column order = weight.reshape(C_out, -1) order."""
    n, c, h, w = t.shape
    u = torch.nn.functional.unfold(t, kernel_size=k, padding=pad)          # (n, c*k*k, h*w)
    return u.permute(0, 2, 1)


class JointTap:
    """Samples h (last-conv input), f (last-BN output) and y (last-act input) at the same positions; accumulates Cov([x; patch(h)]) and Cov(y).
    Bottleneck: last conv is the 1x1 conv3 (patch = h).  Basic: last conv is the 3x3 conv2 (patch = zero-padded 3x3 im2col of h)."""

    def __init__(self, name, blk, positions, gen):
        from layerspec.accumulate import CovarianceAccumulator
        self.name, self.positions, self.gen = name, positions, gen
        if hasattr(blk, "conv3"):
            conv, bn, act = blk.conv3, blk.bn3, blk.act3
        else:
            conv, bn, act = blk.conv2, blk.bn2, blk.act2
        assert conv.stride == (1, 1) and conv.groups == 1, (name, conv)
        self.k, self.pad = conv.kernel_size[0], conv.padding[0]
        C, Ch = conv.out_channels, conv.in_channels * self.k * self.k
        self.C, self.Ch = C, Ch
        self.W3 = conv.weight.detach().double().cpu().numpy().reshape(C, Ch)
        self.joint, self.ycov = CovarianceAccumulator(C + Ch), CovarianceAccumulator(C)
        self.idx = self.h = self.f = None
        self.handles = [conv.register_forward_pre_hook(self._h), bn.register_forward_hook(self._f), act.register_forward_pre_hook(self._y)]

    def _take(self, t):
        n, c, h, w = t.shape
        return t.detach().reshape(n, c, h * w).permute(0, 2, 1)[torch.arange(n)[:, None], self.idx].reshape(-1, c).double().cpu()

    def _h(self, module, inputs):
        t = inputs[0]; n, _, h, w = t.shape; p = min(self.positions, h * w)
        self.idx = torch.rand(n, h * w, generator=self.gen).argsort(1)[:, :p]
        if self.k == 1:
            self.h = self._take(t)
        else:
            cols = im2col(t.detach(), self.k, self.pad)                       # (n, h*w, Ch)
            self.h = cols[torch.arange(n)[:, None], self.idx].reshape(-1, self.Ch).double().cpu()

    def _f(self, module, inputs, output):
        self.f = self._take(output)

    def _y(self, module, inputs):
        y = self._take(inputs[0]); x = y - self.f
        self.joint.update(torch.cat([x, self.h], 1).numpy()); self.ycov.update(y.numpy())

    def cov(self, acc):
        return acc.M2 / (acc.n - 1)


def measure(a):
    import timm
    from layerspec.api import image_loader
    torch.manual_seed(a.seed)
    model = timm.create_model(a.model, pretrained=False, zero_init_last=False).to(a.device).eval()
    gen = torch.Generator().manual_seed(a.seed)
    def is_block(m):
        return (hasattr(m, "conv3") and hasattr(m, "bn3") and hasattr(m, "act3")) or \
               (a.block == "basic" and hasattr(m, "conv2") and hasattr(m, "bn2") and hasattr(m, "act2") and not hasattr(m, "conv3"))
    taps = [JointTap(n, m, a.positions, gen) for n, m in model.named_modules() if is_block(m) and any(n.startswith(s + ".") for s in a.stages)]
    loader = image_loader(a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit)
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(a.device))
    out = {}
    for t in taps:
        out[f"{t.name}::J"] = t.cov(t.joint); out[f"{t.name}::Sy"] = t.cov(t.ycov); out[f"{t.name}::W3"] = t.W3
        out[f"{t.name}::C"] = np.array([t.C]); out[f"{t.name}::n"] = np.array([t.joint.n])
        for h in t.handles:
            h.remove()
    return out


def judge(path, draws=5, seed=0, tau=TAU, thr_a=0.10, thr_b=0.10, need=10):
    z = np.load(path); names = sorted({k.split("::")[0] for k in z.keys()}, key=lambda n: (n.split(".")[0], int(n.split(".")[1])))
    rng = np.random.default_rng(seed); rows = []
    for n in names:
        J, Sy, W3, C = z[f"{n}::J"], z[f"{n}::Sy"], z[f"{n}::W3"], int(z[f"{n}::C"][0])
        std = float(W3.std())
        k_meas = kstar(np.linalg.eigvalsh(Sy), tau); k_rebuilt = kstar(np.linalg.eigvalsh(sigma_y(J, C, W3)), tau)
        k_s, sd_s = sampled_widths(J, C, std, draws, rng, cross=True, tau=tau)
        k_nc, sd_nc = sampled_widths(J, C, std, draws, rng, cross=False, tau=tau)
        Sx, Sh, Cxh = J[:C, :C], J[C:, C:], J[:C, C:]
        rows.append(dict(block=n, C=C, Ch=J.shape[0] - C, n=int(z[f"{n}::n"][0]), std_W3=std, k_meas=k_meas, k_rebuilt=k_rebuilt,
                         k_sampled=k_s, sd_sampled=sd_s, k_nocross=k_nc, sd_nocross=sd_nc,
                         err_sampled=abs(k_s - k_meas) / k_meas, err_nocross=abs(k_nc - k_meas) / k_meas,
                         cross_energy=float((Cxh ** 2).sum() / (np.trace(Sx) * np.trace(Sh)))))
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220):
        print(df.round(3).to_string(index=False))
    san = int((df.k_rebuilt - df.k_meas).abs().le(1).sum())
    a_ = int((df.err_sampled <= thr_a).sum()); b_ = int((df.err_nocross > thr_b).sum()); m = len(df)
    print(f"sanity: rebuilt from J with the network's own last-conv weight equals the measured width (+-1) in {san}/{m}")
    print(f"(a): sampled law within {thr_a:.0%} of measured in {a_}/{m} -> {'HOLDS' if a_ >= need else 'FAILS'} (threshold {need}/{m})")
    print(f"(b): cross-free draws off by > {thr_b:.0%} in {b_}/{m} -> {'HOLDS' if b_ >= need else 'FAILS'} (threshold {need}/{m}); "
          f"off by > 10% in {int((df.err_nocross > 0.10).sum())}/{m}")
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.tv_in1k"); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stages", nargs="+", default=["layer2", "layer3", "layer4"])
    p.add_argument("--data", default="../data/imagenet_val_6400"); p.add_argument("--out", default="../results/addition_law")
    p.add_argument("--judge", action="store_true"); p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=32); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--block", choices=["bottleneck", "basic"], default="bottleneck")
    p.add_argument("--thr-b", type=float, default=0.10); p.add_argument("--need", type=int, default=10)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.model.replace('.', '_')}_s{a.seed}.npz")
    if not a.judge:
        t0 = time.time(); out = measure(a); np.savez_compressed(path, **out); print(f"saved {path} ({time.time() - t0:.0f} s)")
    df = judge(path, a.draws, thr_b=a.thr_b, need=a.need)
    df.to_csv(os.path.join(a.out, f"{a.model.replace('.', '_')}_s{a.seed}_blocks.csv"), index=False)


if __name__ == "__main__":
    main()
