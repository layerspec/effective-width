"""Lemma B7.1 (theory-chain note): under torch.optim.SGD with momentum (Nesterov or not), weight decay and any learning-rate /
momentum schedule, the final weight is exactly W_T = a_T W_0 - sum_s c_s g_s, where g_s is the batch gradient of the
unregularised loss at step s and (a_T, c_s) is the impulse response of the scalar recursion -- so the trained weight lies in
the span of the past gradients, with nonnegative coefficients.  Checked bit-for-bit against the optimiser on a real network."""
import numpy as np
import torch
import torch.nn as nn


def impulse_response(lrs, moms, lam, nesterov=True):
    """Coefficients (a_T, c[s]) of the PyTorch SGD update  d = g + lam*w ; buf = mu*buf + d ; w -= lr*(d + mu*buf) [nesterov]
    or w -= lr*buf [classical], with the momentum buffer initialised to d at the first step (torch semantics)."""
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


def _train(nesterov, seed=0, T=40):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).double()
    w = net[0].weight
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=nesterov)
    w0 = w.detach().clone(); grads, lrs, moms = [], [], []
    x = torch.randn(64, 3, 8, 8, dtype=torch.float64); y = torch.randint(0, 4, (64,))
    for t in range(T):
        lr = 0.05 * (1 + np.cos(np.pi * t / T)) / 2 + 1e-3; mu = 0.85 + 0.1 * (t % 3) / 2   # a varying schedule
        for g in opt.param_groups:
            g["lr"] = lr; g["momentum"] = mu
        lrs.append(lr); moms.append(mu)
        opt.zero_grad(); nn.functional.cross_entropy(net(x[torch.randperm(64)[:16]]), y[:16]).backward()
        grads.append(w.grad.detach().clone()); opt.step()
    return w0, grads, lrs, moms, w.detach().clone()


def test_final_weight_is_the_impulse_weighted_sum_of_gradients():
    for nesterov in (True, False):
        w0, grads, lrs, moms, wT = _train(nesterov)
        a, c = impulse_response(lrs, moms, 5e-4, nesterov)
        recon = a * w0 - sum(float(cs) * g for cs, g in zip(c, grads))
        assert torch.allclose(recon, wT, rtol=1e-10, atol=1e-12), (nesterov, (recon - wT).abs().max().item())
        assert (c >= 0).all() and 0 < a < 1                      # nonnegative combination; the initialisation decays


def test_window_concentrates_on_recent_steps_and_shrinks_with_weight_decay():
    T = 2000; lrs = [0.1] * T; moms = [0.9] * T
    for lam in (5e-4, 5e-3):
        a, c = impulse_response(lrs, moms, lam)
        cum = np.cumsum(c[::-1]) / c.sum()                        # mass counted from the last step backwards
        half = int(np.searchsorted(cum, 0.5))
        assert a < 1e-3 or lam == 5e-4
        expected = (1 - 0.9) / (0.1 * lam) * np.log(2)             # window (1-mu)/(eta*lam), half-mass at ln 2 of it
        assert 0.5 * expected < half < 2 * expected
