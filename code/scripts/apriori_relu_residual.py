#!/usr/bin/env python
"""Where does the Gaussian + ReLU recursion err?  (a-priori step 1b, 2026-09-18)

Hypothesis (scale-mixture proposition, notes/apriori-feasibility-2026-09-18.md): if the pre-activation is
z = s g with g Gaussian and s >= 0 a random scale, E s^2 = 1, then exactly
    Cov(relu z) = Cov_G(relu g) + Var(s) m m^T,   m = E relu g,
i.e. the measured covariance exceeds the Gaussian prediction by a PSD rank-one term along the ReLU mean, and
a rank-one PSD term whose direction has >= tau of its energy in the top-k*(tau) eigenspace cannot raise k*(tau).
This script tests, per ReLU, from the moments saved by apriori_relu_check.py --save-moments:
  neg    fraction of the residual's eigen-mass that is negative   (PSD?)
  r1     lambda_1(D) / sum |lambda(D)|                              (rank one?)
  cos_m  |<u_1(D), m_pred>| / |m_pred|                              (along the mean?)
  align  |P_k m_pred|^2 / |m_pred|^2, k = k*(0.95) of Cov_pred      (lemma condition, need >= 0.95)
  c_hat  least-squares coefficient of m m^T in D; resid = |D - c m m^T| / |D|
  k_fix  k*(0.95) of Cov_pred + c_hat m m^T                          (does one scalar close the gap?)
  kurt   median channel kurtosis of z (raw 4th moment / 2nd^2 - 3); mu/sig = median |E z| / sd z
"""
import sys
import numpy as np
import pandas as pd


def k_star(S, tau=0.95):
    w = np.sort(np.clip(np.linalg.eigvalsh(S), 0, None))[::-1]
    return int(np.searchsorted(np.cumsum(w) / w.sum(), tau) + 1)


def main(path):
    z = np.load(path)
    rows = []
    for key in z["keys"]:
        g = lambda f: z[f"{key}__{f}"]
        Sa, Sp, mp, mz, Sz, m4 = g("Sa"), g("Sp"), g("mp"), g("mz"), g("Sz"), g("m4z")
        C = Sa.shape[0]
        D = Sa - Sp
        lam, U = np.linalg.eigh(D)
        neg = -lam[lam < 0].sum() / np.abs(lam).sum()
        r1 = lam.max() / np.abs(lam).sum()
        u1 = U[:, np.argmax(lam)]
        mh = mp / np.linalg.norm(mp)
        cos_m = abs(u1 @ mh)
        lp, Up = np.linalg.eigh(Sp)
        k = k_star(Sp)
        top = Up[:, np.argsort(lp)[::-1][:k]]
        align = np.linalg.norm(top.T @ mh) ** 2
        c = (mp @ D @ mp) / (mp @ mp) ** 2
        resid = np.linalg.norm(D - c * np.outer(mp, mp)) / np.linalg.norm(D)
        k_fix = k_star(Sp + max(c, 0) * np.outer(mp, mp))
        sd = np.sqrt(np.diag(Sz))
        m2 = np.diag(Sz) + mz ** 2
        kurt = np.median(m4 / m2 ** 2 - 3)
        rows.append(dict(relu=str(key), C=C, k_meas=k_star(Sa), k_pred=k, k_fix=k_fix, neg=neg, r1=r1, cos_m=cos_m,
                         align=align, c_hat=c, resid=resid, kurt=kurt, mu_sig=np.median(np.abs(mz) / sd),
                         tr_ratio=np.trace(D) / np.trace(Sp)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


if __name__ == "__main__":
    main(sys.argv[1])
