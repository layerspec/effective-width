# layerspec 0.2 Public API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the paper's measurement pipeline into an installable package with two functions, `layerspec.profile(model, loader)` and `layerspec.decompose(model, loader)`, that anyone can run on their own CNN.

**Architecture:** Thin wrappers around the existing `hooks.SpectrumProbe` / `metrics` (profile) and the `PatchProbe` core moved out of `scripts/decompose_check.py` (decompose). Each returns a small result object holding a pandas DataFrame with the same columns the paper's CSVs use. The paper's scripts call the same functions, so tool and paper cannot diverge.

**Tech Stack:** Python >= 3.10, PyTorch >= 2.0, numpy, pandas; setuptools build backend; pytest. Optional extras: torchvision, timm.

**Spec:** `docs/superpowers/specs/2026-09-09-layerspec-api-design.md`

## Global Constraints

- Package name `layerspec`, version `0.2.0`, `requires-python = ">=3.10"`.
- Runtime dependencies: `torch>=2.0`, `numpy>=1.24`, `pandas>=2.0` only. `torchvision`/`timm` go in the `models` extra; `pytest` in `dev`.
- `r_max = groups * min((C_in/groups) * k_h * k_w, C_out/groups)` (CLAUDE.md rule 3). Never divide by nominal `C` as the headline statistic.
- Depthwise convolutions are flagged (`is_depthwise`) and excluded from `dense()` (rule 5).
- Layers failing the gate `n >= min_n_over_C * C` are flagged `ok=False`, never dropped.
- No change to the numbers in the paper: the regression in Task 5 must reproduce `results/decompose/resnet50_decompose.csv`.
- All commands below run from `code/` with the project venv: `PY=~/.venvs/effwidth/bin/python`. Tests: `$PY -m pytest tests -q`.
- Commit messages end with the two trailer lines used throughout this repo (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the `Claude-Session:` line).

---

## File structure

| File | Responsibility |
|---|---|
| `code/pyproject.toml` (new) | packaging metadata, dependencies, extras, console script |
| `code/layerspec/__init__.py` (modify) | version 0.2.0; export `profile`, `decompose`, `Profile`, `Decomposition`, `load_model`, `image_loader`, `select_device` |
| `code/layerspec/api.py` (new) | `select_device`, `record_row`, `Profile`, `profile`, `Decomposition`, `decompose`, `load_model`, `image_loader` |
| `code/layerspec/decompose.py` (new) | `PatchProbe`, `kstar`, `erank`, `decompose_rows`, `check_rows` (moved from the script) |
| `code/layerspec/cli.py` (new) | `layerspec profile|decompose` argument parsing, calls the API |
| `code/layerspec/run.py` (modify lines 62-118) | per-record row building replaced by `api.record_row` |
| `code/scripts/decompose_check.py` (modify) | thin wrapper: model/loader setup, then `layerspec.decompose` internals |
| `code/tests/test_api.py` (new) | API tests on a small CNN |
| `README.md`, `code/README.md` (modify) | "Use it on your own model" section |
| `paper/main.tex` Reproducibility (modify) | one sentence: `pip install layerspec` |

---

### Task 1: Packaging skeleton

**Files:**
- Create: `code/pyproject.toml`
- Modify: `code/layerspec/__init__.py`
- Test: existing `code/tests/test_core.py`

**Interfaces:**
- Produces: an installable package; `layerspec.__version__ == "0.2.0"`.

- [ ] **Step 1: Write the failing test**

Append to `code/tests/test_core.py`:

```python
def test_package_version():
    import layerspec
    assert layerspec.__version__ == "0.2.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_core.py::test_package_version -q`
Expected: FAIL, `assert '0.1.0' == '0.2.0'`

- [ ] **Step 3: Create `code/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "layerspec"
version = "0.2.0"
description = "Per-layer effective width of convolutional networks: k*(tau)/r_max with a sample-size gate, and its kernel/data decomposition"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "layerspec"}]
dependencies = [
  "torch>=2.0",
  "numpy>=1.24",
  "pandas>=2.0",
]

[project.optional-dependencies]
models = ["torchvision>=0.15", "timm>=0.9", "pillow>=10.0"]
dev = ["pytest>=7"]

[project.scripts]
layerspec = "layerspec.cli:main"

[project.urls]
Homepage = "https://github.com/layerspec/effective-width"

[tool.setuptools.packages.find]
include = ["layerspec*"]
```

- [ ] **Step 4: Bump the version and install editable**

In `code/layerspec/__init__.py` change `__version__ = "0.1.0"` to `__version__ = "0.2.0"`.

Run: `cd code && ~/.venvs/effwidth/bin/python -m pip install -e ".[dev]" -q`
Expected: installs without error (`layerspec.cli` does not exist yet; the console script is only resolved when invoked, so this is fine).

- [ ] **Step 5: Run the full suite**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests -q`
Expected: all pass (15 tests).

- [ ] **Step 6: Commit**

```bash
git add code/pyproject.toml code/layerspec/__init__.py code/tests/test_core.py
git commit -m "layerspec 0.2.0: pyproject packaging skeleton"
```

---

### Task 2: `record_row`, `Profile` and `profile()`

**Files:**
- Create: `code/layerspec/api.py`
- Modify: `code/layerspec/run.py:62-118` (use `record_row`)
- Modify: `code/layerspec/__init__.py` (exports)
- Test: `code/tests/test_api.py`

**Interfaces:**
- Consumes: `layerspec.hooks.run_probe(model, loader, device, positions_per_image, include_activations, max_batches, seed, progress, pooled, include_blocks, include_bn) -> SpectrumProbe`; `SpectrumProbe.records() -> list[LayerRecord]`; `LayerRecord.meta() -> dict`, `.acc.eigenvalues()`, `.acc.n`, `.C`, `.r_max`, `.acc_pooled`; `layerspec.metrics.compute(eig, C, n, taus) -> SpectrumMetrics` with `.to_row()`.
- Produces:
  - `select_device(device: str | None) -> str`
  - `record_row(rec, min_n_over_C: float, taus: tuple[float, ...]) -> dict | None`
  - `class Profile` with `.table: pd.DataFrame`, `.spectra: dict[str, np.ndarray]`, `.meta: dict`, `.dense(tau=0.95) -> pd.DataFrame`, `.summary(tau=0.95) -> dict`, `.to_csv(path)`, `Profile.from_csv(path)`.
  - `profile(model, loader, *, device=None, positions=16, taus=(0.9, 0.95, 0.99, 0.999), min_n_over_C=50.0, include_activations=True, include_blocks=True, include_bn=False, max_batches=None, seed=0, progress=False) -> Profile`

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_api.py`:

```python
"""Tests for the public API (layerspec.profile / layerspec.decompose)."""
import numpy as np
import pytest
import torch
import torch.nn as nn


def small_cnn(seed: int = 0) -> nn.Module:
    """Dense 3x3, a 1x1 expansion (r_max binds), a depthwise 3x3, a 1x1 reduction."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 64, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.ReLU(),
        nn.Conv2d(64, 8, 1),
    )


def random_loader(n_images: int = 64, batch: int = 16, size: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_images, 3, size, size, generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, torch.zeros(n_images, dtype=torch.long)),
                                       batch_size=batch, shuffle=False)


def test_profile_table_columns_and_bound():
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(), device="cpu", positions=16)
    t = prof.table
    for col in ["layer", "kind", "depth_index", "C", "groups", "is_depthwise", "r_max",
                "n_samples", "n_over_C", "ok", "k_star_0.95", "k_star_rmax_0.95",
                "k_star_rmax_0.999", "participation_ratio", "effective_rank", "stable_rank"]:
        assert col in t.columns, col
    conv = t[t.kind == "conv"]
    assert len(conv) == 4
    assert (conv["k_star_rmax_0.999"] <= 1.0 + 1e-9).all()
    assert list(conv.r_max) == [16, 16, 64, 8]
    assert list(conv.is_depthwise) == [False, False, True, False]


def test_profile_gate_flags_but_keeps_rows():
    import layerspec
    # 4 images x 4 positions = 16 samples: every layer fails n/C >= 50
    prof = layerspec.profile(small_cnn(), random_loader(n_images=4, batch=4), device="cpu", positions=4)
    conv = prof.table[prof.table.kind == "conv"]
    assert len(conv) == 4
    assert not conv.ok.any()
    assert prof.dense().empty
    # 64 images x 16 positions = 1024 samples: the 16-channel layer passes (n/C = 64), the 64-channel ones do not
    prof = layerspec.profile(small_cnn(), random_loader(), device="cpu", positions=16)
    conv = prof.table[prof.table.kind == "conv"]
    assert conv.ok.tolist() == [True, False, False, True]
    d = prof.dense(tau=0.95)
    assert list(d.columns) == ["layer", "depth_index", "r_max", "k_star_rmax_0.95"]
    assert len(d) == 2                      # dense + ok; the depthwise layer is out even when ok


def test_profile_summary_and_csv_roundtrip(tmp_path):
    import layerspec
    prof = layerspec.profile(small_cnn(), random_loader(n_images=256), device="cpu", positions=16)
    s = prof.summary(tau=0.95)
    assert set(s) == {"L", "median_level", "rho_depth", "rmax_check_max", "rmax_check_ok"}
    assert s["rmax_check_ok"] is True and s["L"] == 3
    path = tmp_path / "p.csv"
    prof.to_csv(path)
    back = layerspec.Profile.from_csv(path)
    a = prof.table.sort_values(["depth_index", "kind"]).reset_index(drop=True)
    b = back.table.sort_values(["depth_index", "kind"]).reset_index(drop=True)
    assert np.allclose(a["k_star_rmax_0.95"].astype(float), b["k_star_rmax_0.95"].astype(float), equal_nan=True)
    import pandas as pd
    pd.testing.assert_frame_equal(back.dense(), prof.dense(), check_dtype=False)


def test_profile_rejects_models_without_conv():
    import layerspec
    with pytest.raises(ValueError):
        layerspec.profile(nn.Sequential(nn.Flatten(), nn.Linear(192, 4)), random_loader(), device="cpu")


def test_select_device_prefers_available_accelerator():
    from layerspec.api import select_device
    assert select_device("cpu") == "cpu"
    auto = select_device(None)
    assert auto in ("cuda", "mps", "cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_api.py -q`
Expected: FAIL / ERROR with `AttributeError: module 'layerspec' has no attribute 'profile'` (or ImportError for `layerspec.api`).

- [ ] **Step 3: Write `code/layerspec/api.py` (profile half)**

```python
"""Public API: profile(model, loader) and decompose(model, loader).

Both are thin wrappers over the measurement code the paper used
(hooks.SpectrumProbe, metrics, decompose.PatchProbe); the paper's own
scripts call the same functions, so the released numbers and this tool
cannot diverge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import metrics as _metrics
from .hooks import run_probe

DEFAULT_TAUS = (0.9, 0.95, 0.99, 0.999)


def select_device(device: str | None) -> str:
    """Explicit device, else cuda > mps > cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tau_key(tau: float) -> str:
    return f"{tau:g}"


def record_row(rec, min_n_over_C: float, taus: tuple[float, ...] = DEFAULT_TAUS) -> dict | None:
    """One table row for a hooked tensor, or None if it has too few samples to
    form a covariance at all.  This is the row layout of results/*_layers.csv."""
    if rec.acc is None or rec.acc.n < 2:
        return None
    eig = rec.acc.eigenvalues()
    row = rec.meta()
    row.update(_metrics.compute(eig, rec.C, rec.acc.n, taus=taus).to_row())
    row["n_over_C"] = rec.acc.n / rec.C
    # A sample covariance from n < C samples is rank-deficient by construction;
    # below n/C = 50 the small eigenvalues are still biased.  Flag, never drop.
    row["n_over_C_ok"] = bool(rec.acc.n >= min_n_over_C * rec.C)
    if rec.r_max:
        for tau in taus:
            k = row.get(f"k_star_{_tau_key(tau)}")
            if k is not None:
                row[f"k_star_rmax_{_tau_key(tau)}"] = k / rec.r_max
        row["r_max_over_C"] = rec.r_max / rec.C
    if rec.acc_pooled is not None and rec.acc_pooled.n >= 2:
        pm = _metrics.compute(rec.acc_pooled.eigenvalues(), rec.C, rec.acc_pooled.n, taus=taus).to_row()
        for k, v in pm.items():
            if k != "C":
                row[f"pooled_{k}"] = v
        row["pooled_n_over_C"] = rec.acc_pooled.n / rec.C
        row["pooled_n_over_C_ok"] = bool(rec.acc_pooled.n >= min_n_over_C * rec.C)
    return row


@dataclass
class Profile:
    """Per-layer effective-width profile of one network on one image set."""
    table: pd.DataFrame
    spectra: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def dense(self, tau: float = 0.95) -> pd.DataFrame:
        """Dense convolutions that pass the sample-size gate, in depth order:
        the rows the paper's profiles are made of."""
        col = f"k_star_rmax_{_tau_key(tau)}"
        t = self.table
        keep = (t.kind == "conv") & t["ok"].astype(bool) & ~t["is_depthwise"].astype(bool)
        return (t.loc[keep, ["layer", "depth_index", "r_max", col]]
                .sort_values("depth_index").reset_index(drop=True))

    def summary(self, tau: float = 0.95) -> dict:
        d = self.dense(tau)
        col = f"k_star_rmax_{_tau_key(tau)}"
        conv = self.table[(self.table.kind == "conv") & self.table["k_star_rmax_0.999"].notna()]
        rmax_max = float(conv["k_star_rmax_0.999"].max()) if len(conv) else float("nan")
        rho = float("nan")
        if len(d) >= 3:
            from scipy.stats import spearmanr  # optional; scipy ships with pandas' extras on most installs
            rho = float(spearmanr(np.arange(len(d)), d[col].to_numpy()).statistic)
        return {"L": int(len(d)),
                "median_level": float(d[col].median()) if len(d) else float("nan"),
                "rho_depth": rho,
                "rmax_check_max": rmax_max,
                "rmax_check_ok": bool(rmax_max <= 1.0 + 1e-9) if len(conv) else True}

    def to_csv(self, path) -> None:
        self.table.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path) -> "Profile":
        t = pd.read_csv(path)
        if "ok" not in t.columns and "n_over_C_ok" in t.columns:
            t["ok"] = t["n_over_C_ok"].astype(bool)
        return cls(table=t, meta={"source": os.fspath(path)})


def _has_conv(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


@torch.no_grad()
def profile(model: nn.Module, loader, *, device: str | None = None, positions: int = 16,
            taus: tuple[float, ...] = DEFAULT_TAUS, min_n_over_C: float = 50.0,
            include_activations: bool = True, include_blocks: bool = True, include_bn: bool = False,
            max_batches: int | None = None, seed: int = 0, progress: bool = False) -> Profile:
    """Measure k*(tau)/r_max (and PR, effective rank, stable rank) for every
    convolution output of `model` over the images `loader` yields.

    `loader` may yield tensors or (x, y, ...) tuples; the first element is the
    image batch.  Every hooked tensor gets a row; `Profile.dense()` applies the
    paper's gate and depthwise exclusion.
    """
    if not _has_conv(model):
        raise ValueError("profile() needs a model with at least one nn.Conv2d")
    device = select_device(device)
    probe = run_probe(model, loader, device=device, positions_per_image=positions,
                      include_activations=include_activations, max_batches=max_batches,
                      seed=seed, progress=progress, pooled=True, include_blocks=include_blocks,
                      include_bn=include_bn)
    rows, spectra = [], {}
    for rec in probe.records():
        row = record_row(rec, min_n_over_C, taus)
        if row is None:
            continue
        row["ok"] = row["n_over_C_ok"]
        rows.append(row)
        spectra[f"{rec.name}::{rec.kind}#{rec.call_index}"] = rec.acc.eigenvalues()
    table = pd.DataFrame(rows)
    meta = {"device": device, "positions_per_image": positions, "seed": seed,
            "min_n_over_C": min_n_over_C, "torch": torch.__version__, "n_rows": len(table)}
    return Profile(table=table, spectra=spectra, meta=meta)
```

Note on `summary()`: `scipy` is not a declared dependency. Replace the `spearmanr` import with a local rank correlation so the package stays at three dependencies:

```python
def _spearman(a, b) -> float:
    a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
```

and in `summary()` use `rho = _spearman(np.arange(len(d)), d[col].to_numpy())` when `len(d) >= 3`. (Ties are handled by average ranks, the same convention as scipy's `spearmanr`.)

- [ ] **Step 4: Export from `__init__.py`**

Replace the two import lines at the bottom of `code/layerspec/__init__.py` with:

```python
from .accumulate import CovarianceAccumulator, merge  # noqa: F401
from . import metrics  # noqa: F401
from .api import (profile, Profile, select_device)  # noqa: F401
```

(`decompose`, `Decomposition`, `load_model`, `image_loader` are added in Tasks 3 and 6.)

- [ ] **Step 5: Run the API tests**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_api.py -q`
Expected: 5 passed.

- [ ] **Step 6: Make `run.py` use `record_row`**

In `code/layerspec/run.py`, replace the body of the `for rec in probe.records():` loop from `if rec.acc is None or rec.acc.n < 2:` down to `layer_rows.append(row)` (lines 65-113) with:

```python
    for rec in probe.records():
        row = record_row(rec, args.min_n_over_C, _metrics.DEFAULT_TAUS)
        if row is None:
            continue
        spectra[f"{rec.name}::{rec.kind}"] = rec.acc.eigenvalues()
        row["model"] = model_name
        row["weights"] = weights_tag
        row["source"] = source
        layer_rows.append(row)
```

and add `from .api import record_row` next to the other relative imports at the top of `run.py`. Keep the `for mult, ck in rec.checkpoints.items():` loop that follows (it is inside the same `for rec` loop; move it under the new body).

- [ ] **Step 7: Verify the column set of the paper's CSVs is unchanged**

Run:
```bash
cd code && ~/.venvs/effwidth/bin/python -m layerspec.run --data synthetic --models resnet18_random --device cpu --limit 32 --batch-size 16 --workers 0 --out /tmp/ls_check --positions 4
~/.venvs/effwidth/bin/python - <<'EOF'
import pandas as pd
new = set(pd.read_csv("/tmp/ls_check/resnet18_random_layers.csv").columns)
old = set(pd.read_csv("../results/local6400/trained/resnet50_layers.csv").columns)
print("missing:", sorted(old - new)); print("extra:", sorted(new - old))
EOF
```
Expected: `missing: []`. (`extra` may list nothing; if it lists columns, remove them from `record_row` until it is empty.)

- [ ] **Step 8: Run the full suite and commit**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests -q`
Expected: all pass.

```bash
git add code/layerspec/api.py code/layerspec/__init__.py code/layerspec/run.py code/tests/test_api.py
git commit -m "layerspec.profile(): public API over SpectrumProbe; run.py shares record_row"
```

---

### Task 3: `layerspec/decompose.py` (core moved out of the script) and `decompose()`

**Files:**
- Create: `code/layerspec/decompose.py`
- Modify: `code/layerspec/api.py` (add `Decomposition`, `decompose`)
- Modify: `code/layerspec/__init__.py` (exports)
- Test: `code/tests/test_api.py`

**Interfaces:**
- Consumes: `layerspec.accumulate.CovarianceAccumulator(dim)` with `.update_precomputed(m, mean, M2)`, `.covariance`, `.n`.
- Produces:
  - `decompose.kstar(eig, tau) -> int`, `decompose.erank(eig) -> float`
  - `decompose.PatchProbe(model, positions, seed)` with `.acc: dict[name, CovarianceAccumulator]`, `.meta: dict[name, dict]`, `.remove()`
  - `decompose.decompose_rows(probe, taus, rank_tol) -> list[dict]`
  - `decompose.check_rows(df, taus) -> dict`
  - `api.Decomposition` with `.table`, `.meta`, `.check() -> dict`, `.to_csv(path)`
  - `api.decompose(model, loader, *, device=None, positions=48, taus=DEFAULT_TAUS, rank_tol=1e-6, max_batches=None, seed=0) -> Decomposition`

- [ ] **Step 1: Write the failing tests**

Append to `code/tests/test_api.py`:

```python
def test_decompose_columns_and_propositions():
    import layerspec
    dec = layerspec.decompose(small_cnn(), random_loader(n_images=256), device="cpu", positions=16)
    t = dec.table
    for col in ["layer", "depth_index", "C_in", "C_out", "k", "d", "r_max", "n_samples", "n_over_d",
                "kernel_erank_over_rmax", "kernel_cond_top_rmax", "data_erank_over_rmax",
                "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95", "ostrowski_ok", "kappa"]:
        assert col in t.columns, col
    assert len(t) == 3                                   # the depthwise layer (groups=64) is not dense
    assert t.ostrowski_ok.all()
    for tau in (0.9, 0.95, 0.99, 0.999):
        assert t[f"bound_ok_{tau}"].all()               # corollary: k*_out(tau) <= k*_ortho(tau')
    c = dec.check()
    assert c["ostrowski"] == (3, 3) and c["kappa_median"] > 0


def test_decompose_identity_matches_direct_measurement():
    """Sigma_out = W Sigma_patch W^T: the spectrum recomputed from the kernel
    and the patch covariance agrees with the directly accumulated output
    spectrum on a 1x1 layer (patch == input pixel), up to sampling noise."""
    import layerspec
    from layerspec.hooks import SpectrumProbe
    torch.manual_seed(1)
    net = nn.Sequential(nn.Conv2d(16, 8, 1))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2000, 16, 8, 8, generator=torch.Generator().manual_seed(3))),
        batch_size=100)
    dec = layerspec.decompose(net, loader, device="cpu", positions=16, seed=0)
    probe = SpectrumProbe(net, positions_per_image=16, pooled=False, include_activations=False, seed=0)
    with torch.no_grad():
        for (x,) in loader:
            net(x)
    rec = [r for r in probe.records() if r.kind == "conv"][0]
    probe.remove()
    lam_direct = np.sort(rec.acc.eigenvalues())[::-1]
    W = net[0].weight.detach().double().reshape(8, 16).numpy()
    Sig = dec.patch_covariance["0"]
    lam_recomputed = np.sort(np.linalg.eigvalsh(W @ Sig @ W.T))[::-1]
    assert np.allclose(lam_direct / lam_direct.sum(), lam_recomputed / lam_recomputed.sum(), atol=0.02)


def test_decompose_rejects_models_without_dense_conv():
    import layerspec
    with pytest.raises(ValueError):
        layerspec.decompose(nn.Sequential(nn.Conv2d(4, 4, 3, groups=4)), random_loader(), device="cpu")
```

(`Decomposition.patch_covariance` is a `{layer: Sigma_patch ndarray}` dict, added below so the identity can be checked from outside.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_api.py -q -k decompose`
Expected: FAIL with `AttributeError: module 'layerspec' has no attribute 'decompose'`.

- [ ] **Step 3: Create `code/layerspec/decompose.py`**

```python
"""Where a layer's effective width comes from: the kernel or the data.

At the conv output (before the nonlinearity and the shortcut)
    Sigma_out = W Sigma_patch W^T,
W the (C_out x d) kernel matrix, d = C_in*k_h*k_w, Sigma_patch the d x d
covariance of the receptive-field patch (im2col column).  PatchProbe
accumulates Sigma_patch per dense convolution; decompose_rows reports, per
layer and tau,
    out      k*(W Sigma W^T)/r_max     the measured quantity, recomputed
    kernel   k*(W W^T)/r_max           the layer under white patches
    data     k*(Sigma_patch)/r_max     the patch covariance, capped at r_max
    ortho    k*(W_o Sigma W_o^T)/r_max W_o the polar factor of W: what
                                        orthogonalising this kernel would give with this data
and the Proposition-1 checks (Ostrowski ratios, corollary bound).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .accumulate import CovarianceAccumulator

DEFAULT_TAUS = (0.9, 0.95, 0.99, 0.999)


def kstar(eig: np.ndarray, tau: float) -> int:
    eig = np.clip(eig, 0, None)
    if eig.sum() <= 0:
        return 0
    c = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(c, tau) + 1)


def erank(eig: np.ndarray) -> float:
    eig = np.clip(eig, 0, None)
    p = eig / eig.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


class PatchProbe:
    """Forward-pre-hooks on every dense (groups == 1) nn.Conv2d that accumulate
    the covariance of `positions` sampled im2col columns per image."""

    def __init__(self, model: nn.Module, positions: int, seed: int):
        self.acc: dict[str, CovarianceAccumulator] = {}
        self.meta: dict[str, dict] = {}
        self.handles = []
        self.p = positions
        self.gen = torch.Generator().manual_seed(seed)
        order = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                self.meta[name] = dict(depth_index=order, module=m)
                order += 1
                self.handles.append(m.register_forward_pre_hook(
                    lambda mod, inp, _n=name: self._consume(_n, mod, inp[0])))

    @torch.no_grad()
    def _consume(self, name: str, m: nn.Conv2d, x: torch.Tensor) -> None:
        N = x.shape[0]
        cols = F.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride)  # (N, d, L)
        d, L = cols.shape[1], cols.shape[2]
        if L == 1:
            return                                   # conv on a pooled tensor: not in the profile
        p = min(self.p, L)
        idx = torch.randint(0, L, (N, p), generator=self.gen).to(cols.device)
        cols = torch.gather(cols, 2, idx.unsqueeze(1).expand(N, d, p))   # (N, d, p)
        xs = cols.permute(0, 2, 1).reshape(N * p, d).to(torch.float32)
        acc = self.acc.get(name)
        if acc is None:
            acc = self.acc[name] = CovarianceAccumulator(d)
            self.meta[name].update(d=d, C_in=m.in_channels, C_out=m.out_channels,
                                   k=m.kernel_size[0] * m.kernel_size[1])
        bm = xs.mean(0)
        xc = xs - bm
        acc.update_precomputed(xs.shape[0], bm.cpu().double().numpy(), (xc.T @ xc).cpu().double().numpy())

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _sym_eigs(S: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]


def decompose_rows(probe: PatchProbe, taus: tuple[float, ...] = DEFAULT_TAUS,
                   rank_tol: float = 1e-6) -> list[dict]:
    """One row per dense convolution with an accumulated patch covariance."""
    rows = []
    for name, acc in probe.acc.items():
        meta = probe.meta[name]
        m = meta["module"]
        d, C_out = meta["d"], meta["C_out"]
        r_max = min(d, C_out)
        W = m.weight.detach().cpu().double().reshape(C_out, d).numpy()
        Sig = acc.covariance
        lam_data = np.sort(np.linalg.eigvalsh(Sig))[::-1]
        lam_out = _sym_eigs(W @ Sig @ W.T)
        lam_kernel = _sym_eigs(W @ W.T)
        U, sv, Vt = np.linalg.svd(W, full_matrices=False)
        Wo = U @ Vt                                     # polar factor, nearest (semi-)isometry
        lam_ortho = _sym_eigs(Wo @ Sig @ Wo.T)
        # Proposition 1: with W = P W_o, Sigma_out = P Sigma_o P and (Ostrowski)
        # lambda_i(out) = theta_i lambda_i(ortho), sigma_min^2 <= theta_i <= sigma_max^2
        # over the r = rank(W) nonzero directions; hence
        #   k*_out(tau) <= k*_ortho(tau'),  tau' = kappa^2 tau / (1 - tau + kappa^2 tau).
        r = int((sv > sv[0] * rank_tol).sum())
        kappa = float(sv[0] / sv[r - 1])
        n_null = int(len(sv) - r)
        ratio = lam_out[:r] / np.clip(lam_ortho[:r], 1e-300, None)
        ostrowski_ok = bool(np.all(ratio >= sv[r - 1] ** 2 * (1 - 1e-6))
                            and np.all(ratio <= sv[0] ** 2 * (1 + 1e-6)))
        row = dict(layer=name, depth_index=meta["depth_index"], C_in=meta["C_in"], C_out=C_out,
                   k=meta["k"], d=d, r_max=r_max, n_samples=acc.n, n_over_d=acc.n / d,
                   kernel_erank_over_rmax=erank(sv ** 2) / r_max,
                   kernel_cond_top_rmax=float(sv[0] / sv[min(r_max, len(sv)) - 1]),
                   data_erank_over_rmax=min(erank(lam_data), r_max) / r_max,
                   rank_W=r, n_null=n_null, kappa=kappa, ostrowski_ok=ostrowski_ok)
        for tau in taus:
            tau_k = kappa ** 2 * tau / (1 - tau + kappa ** 2 * tau)
            row[f"bound_{tau}"] = kstar(lam_ortho, tau_k) / r_max if tau_k < 1 else 1.0
            row[f"bound_ok_{tau}"] = bool(kstar(lam_out, tau) <= (kstar(lam_ortho, tau_k) if tau_k < 1 else r_max))
            row[f"out_{tau}"] = kstar(lam_out, tau) / r_max
            row[f"kernel_{tau}"] = kstar(lam_kernel, tau) / r_max
            row[f"data_{tau}"] = min(kstar(lam_data, tau), r_max) / r_max
            row[f"ortho_{tau}"] = kstar(lam_ortho, tau) / r_max
        rows.append(row)
    return rows


def check_rows(df: pd.DataFrame, taus: tuple[float, ...] = DEFAULT_TAUS) -> dict:
    """Proposition-1 counts over a decomposition table."""
    n = len(df)
    return {"ostrowski": (int(df.ostrowski_ok.sum()), n),
            "corollary": {tau: (int(df[f"bound_ok_{tau}"].sum()), n) for tau in taus},
            "kappa_median": float(df.kappa.median()) if n else float("nan"),
            "kappa_max": float(df.kappa.max()) if n else float("nan"),
            "n_null_layers": int((df.n_null > 0).sum())}


def format_check(c: dict) -> str:
    """The one-line summary decompose_check.py prints (kept byte-identical)."""
    return (f"Proposition 1 check: Ostrowski ratios within [s_min^2, s_max^2] on {c['ostrowski'][0]}/{c['ostrowski'][1]} layers; "
            f"k*_out(tau) <= k*_ortho(tau') on " + ", ".join(f"tau={t}: {v[0]}/{v[1]}" for t, v in c["corollary"].items())
            + f";  kappa median {c['kappa_median']:.1f}, max {c['kappa_max']:.1f}")
```

- [ ] **Step 4: Add `Decomposition` and `decompose()` to `code/layerspec/api.py`**

Append:

```python
from . import decompose as _dec


@dataclass
class Decomposition:
    """Kernel/data decomposition of the effective width, one row per dense convolution."""
    table: pd.DataFrame
    patch_covariance: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def check(self) -> dict:
        return _dec.check_rows(self.table, tuple(self.meta.get("taus", _dec.DEFAULT_TAUS)))

    def to_csv(self, path) -> None:
        self.table.to_csv(path, index=False)


@torch.no_grad()
def decompose(model: nn.Module, loader, *, device: str | None = None, positions: int = 48,
              taus: tuple[float, ...] = DEFAULT_TAUS, rank_tol: float = 1e-6,
              max_batches: int | None = None, seed: int = 0) -> Decomposition:
    """Accumulate the receptive-field patch covariance of every dense
    convolution and split k*/r_max into out / kernel / data / ortho."""
    if not any(isinstance(m, nn.Conv2d) and m.groups == 1 for m in model.modules()):
        raise ValueError("decompose() needs at least one dense (groups == 1) nn.Conv2d")
    device = select_device(device)
    model = model.to(device).eval()
    probe = _dec.PatchProbe(model, positions, seed)
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))
    finally:
        probe.remove()
    rows = _dec.decompose_rows(probe, taus, rank_tol)
    table = pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)
    cov = {name: acc.covariance for name, acc in probe.acc.items()}
    meta = {"device": device, "positions": positions, "seed": seed, "rank_tol": rank_tol,
            "taus": tuple(taus), "torch": torch.__version__}
    return Decomposition(table=table, patch_covariance=cov, meta=meta)
```

Then in `code/layerspec/__init__.py` change the api import line to:

```python
from .api import (profile, Profile, decompose, Decomposition, select_device)  # noqa: F401
```

- [ ] **Step 5: Run the tests**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests -q`
Expected: all pass (8 in test_api + 15 in test_core).

- [ ] **Step 6: Commit**

```bash
git add code/layerspec/decompose.py code/layerspec/api.py code/layerspec/__init__.py code/tests/test_api.py
git commit -m "layerspec.decompose(): PatchProbe and the four quantities moved into the package"
```

---

### Task 4: `scripts/decompose_check.py` becomes a wrapper

**Files:**
- Modify: `code/scripts/decompose_check.py` (replace `PatchProbe`, `kstar`, `erank` and the row loop with package calls; keep the CLI flags and output format)

**Interfaces:**
- Consumes: `layerspec.decompose.PatchProbe`, `decompose_rows`, `check_rows`, `format_check`.
- Produces: the same CSV file and the same printed lines as before.

- [ ] **Step 1: Rewrite the script**

Replace everything from `TAUS = [0.9, 0.95, 0.99, 0.999]` down to (but not including) `def main(argv=None) -> int:` with:

```python
from layerspec.decompose import PatchProbe, decompose_rows, check_rows, format_check   # noqa: E402

TAUS = (0.9, 0.95, 0.99, 0.999)
```

and replace the body of `main()` from `model = model.to(a.device).eval()` to the end of the function with:

```python
    model = model.to(a.device).eval()
    probe = PatchProbe(model, a.positions, a.seed)
    t0 = time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(a.device))
    probe.remove()
    print(f"{a.model} ({tag}): {len(probe.acc)} dense conv layers, {time.time() - t0:.0f}s", flush=True)

    rows = decompose_rows(probe, TAUS, a.rank_tol)
    for r in rows:
        r["model"] = a.model
    df = pd.DataFrame(rows).sort_values("depth_index")
    df = df[["model"] + [c for c in df.columns if c != "model"]]
    path = os.path.join(a.out, f"{a.model.replace(':', '_')}_decompose.csv")
    df.to_csv(path, index=False)
    print(df[["layer", "d", "C_out", "n_over_d", "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95",
              "kernel_erank_over_rmax"]].to_string(index=False), flush=True)
    print(format_check(check_rows(df, TAUS)), flush=True)
    print("wrote", path, flush=True)
    return 0
```

Delete the now-unused imports (`numpy as np`, `torch.nn as nn`, `torch.nn.functional as F`, `CovarianceAccumulator`) from the script header; keep `argparse, os, sys, time, pandas, torch`, `build_loader`, `build`.

- [ ] **Step 2: Smoke-run on synthetic data**

Run: `cd code && ~/.venvs/effwidth/bin/python scripts/decompose_check.py --model resnet18_random --data synthetic --device cpu --limit 32 --batch-size 16 --workers 0 --positions 4 --out /tmp/ls_dec --name smoke`
Expected: prints the layer table and a `Proposition 1 check:` line, writes `/tmp/ls_dec/smoke_decompose.csv`. Check the column set:

```bash
~/.venvs/effwidth/bin/python -c "
import pandas as pd
new=list(pd.read_csv('/tmp/ls_dec/smoke_decompose.csv').columns); old=list(pd.read_csv('../results/decompose/resnet50_decompose.csv').columns)
print('same columns, same order:', new==old); print(set(old)^set(new))"
```
Expected: `same columns, same order: True` and an empty set.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/decompose_check.py
git commit -m "decompose_check.py: thin wrapper over layerspec.decompose"
```

---

### Task 5: Regression against the paper's ResNet-50 decomposition (run once)

**Files:** none modified. Output goes to the scratch directory.

- [ ] **Step 1: Re-run the ResNet-50 decomposition with the wrapper**

Run (Mac, MPS, about 10-15 minutes; use `nohup caffeinate` so it survives idle):
```bash
cd code && nohup caffeinate -i ~/.venvs/effwidth/bin/python scripts/decompose_check.py --model resnet50 --data ../data/imagenet_val_6400 --device mps --out /tmp/ls_regress > /tmp/ls_regress.log 2>&1 &
```
Wait until `/tmp/ls_regress.log` ends with `wrote /tmp/ls_regress/resnet50_decompose.csv`.

- [ ] **Step 2: Compare row for row**

```bash
~/.venvs/effwidth/bin/python - <<'EOF'
import pandas as pd, numpy as np
a = pd.read_csv("../results/decompose/resnet50_decompose.csv").set_index("layer")
b = pd.read_csv("/tmp/ls_regress/resnet50_decompose.csv").set_index("layer")
assert list(a.index) == list(b.index), "layer order differs"
num = [c for c in a.columns if a[c].dtype.kind in "fi"]
bad = [c for c in num if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), rtol=1e-6, atol=1e-9, equal_nan=True)]
print("mismatching numeric columns:", bad)
print("ostrowski_ok identical:", (a.ostrowski_ok == b.ostrowski_ok).all())
EOF
```
Expected: `mismatching numeric columns: []` and `True`. If `kappa`/`kernel_cond_top_rmax` differ in the last digits only (MPS float32 nondeterminism in the forward pass), loosen to `rtol=1e-4` for those two columns and record that in the commit message; any difference in `out_*`, `kernel_*`, `data_*`, `ortho_*` means the move changed the computation and must be fixed before continuing.

- [ ] **Step 3: Record the regression in the notes**

Append to `notes/status-zh.md` under the round-3 section: `- layerspec 0.2：decompose 搬進套件後在 ResNet-50 6,400 張子集重跑，與 results/decompose/resnet50_decompose.csv 逐層一致（2026-09-09）`.

```bash
git add notes/status-zh.md
git commit -m "Notes: decompose regression against the paper's ResNet-50 table passes"
```

---

### Task 6: `load_model`, `image_loader` and the CLI

**Files:**
- Modify: `code/layerspec/api.py` (add `load_model`, `image_loader`)
- Create: `code/layerspec/cli.py`
- Modify: `code/layerspec/__init__.py` (exports)
- Test: `code/tests/test_api.py`

**Interfaces:**
- Consumes: `layerspec.models.build(name, pretrained=True, seed=0) -> (nn.Module, str)`; `layerspec.data.build_loader(data, batch_size, num_workers, image_size, limit, seed, preprocess) -> (DataLoader, str)`.
- Produces:
  - `load_model(name: str, *, pretrained=True, seed=0) -> nn.Module` (names as in `models.build`: torchvision registry names, `timm:<name>`, `<name>_random`)
  - `image_loader(path: str | None, *, batch_size=32, workers=4, image_size=224, limit=None, seed=0, preprocess="crop") -> DataLoader`
  - `cli.main(argv=None) -> int` with subcommands `profile` and `decompose`.

- [ ] **Step 1: Write the failing tests**

Append to `code/tests/test_api.py`:

```python
def test_load_model_and_image_loader_synthetic():
    import layerspec
    m = layerspec.load_model("resnet18_random", seed=0)
    assert isinstance(m, nn.Module)
    loader = layerspec.image_loader(None, batch_size=4, limit=8, image_size=32)
    x = next(iter(loader))
    x = x[0] if isinstance(x, (list, tuple)) else x
    assert tuple(x.shape) == (4, 3, 32, 32)


def test_cli_profile_writes_csv(tmp_path):
    from layerspec.cli import main
    out = tmp_path / "prof.csv"
    rc = main(["profile", "--model", "resnet18_random", "--data", "synthetic", "--limit", "8",
               "--image-size", "32", "--batch-size", "4", "--positions", "4", "--device", "cpu",
               "--out", str(out)])
    assert rc == 0 and out.exists()
    import pandas as pd
    t = pd.read_csv(out)
    assert "k_star_rmax_0.95" in t.columns and (t.kind == "conv").sum() == 20


def test_cli_decompose_writes_csv(tmp_path):
    from layerspec.cli import main
    out = tmp_path / "dec.csv"
    rc = main(["decompose", "--model", "resnet18_random", "--data", "synthetic", "--limit", "8",
               "--image-size", "32", "--batch-size", "4", "--positions", "4", "--device", "cpu",
               "--out", str(out)])
    assert rc == 0 and out.exists()
    import pandas as pd
    assert "ortho_0.95" in pd.read_csv(out).columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_api.py -q -k "load_model or cli"`
Expected: FAIL (`load_model` missing; `layerspec.cli` missing).

- [ ] **Step 3: Add the conveniences to `code/layerspec/api.py`**

Append:

```python
def load_model(name: str, *, pretrained: bool = True, seed: int = 0) -> nn.Module:
    """A torchvision model by registry name ("resnet50", "vgg16_bn", ...), a timm
    checkpoint ("timm:resnet50.a1_in1k"), or either with the suffix "_random"
    for the same architecture at initialisation.  Needs the [models] extra."""
    from .models import build
    model, _tag = build(name, pretrained=pretrained, seed=seed)
    return model


def image_loader(path: str | None, *, batch_size: int = 32, workers: int = 4, image_size: int = 224,
                 limit: int | None = None, seed: int = 0, preprocess: str = "crop"):
    """A DataLoader over every image under `path` (any directory layout; labels
    are not needed), with ImageNet normalisation.  `path=None` or "synthetic"
    gives 1/f-noise images for smoke tests.  Needs the [models] extra for real images."""
    from .data import build_loader
    loader, _src = build_loader(path, batch_size=batch_size, num_workers=workers, image_size=image_size,
                                limit=limit, seed=seed, preprocess=preprocess)
    return loader
```

Update the `__init__.py` api import line to:

```python
from .api import (profile, Profile, decompose, Decomposition, select_device,  # noqa: F401
                  load_model, image_loader)
```

- [ ] **Step 4: Create `code/layerspec/cli.py`**

```python
"""Command line: layerspec profile|decompose --model NAME --data DIR --out FILE.csv"""
from __future__ import annotations

import argparse
import sys

from . import api


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help='torchvision name, "timm:<name>", or "<name>_random"')
    p.add_argument("--data", default="synthetic", help="image directory (any layout) or 'synthetic'")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit", type=int, default=None, help="use at most this many images")
    p.add_argument("--seed", type=int, default=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="layerspec", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("profile", help="k*(tau)/r_max for every convolution output")
    _common(pp)
    pp.add_argument("--positions", type=int, default=16, help="sampled spatial positions per image")
    pp.add_argument("--min-n-over-C", type=float, default=50.0)
    pd_ = sub.add_parser("decompose", help="out / kernel / data / ortho per dense convolution")
    _common(pd_)
    pd_.add_argument("--positions", type=int, default=48)
    pd_.add_argument("--rank-tol", type=float, default=1e-6)
    a = p.parse_args(argv)

    model = api.load_model(a.model, seed=a.seed)
    loader = api.image_loader(None if a.data == "synthetic" else a.data, batch_size=a.batch_size,
                              workers=a.workers, image_size=a.image_size, limit=a.limit, seed=a.seed)
    if a.cmd == "profile":
        res = api.profile(model, loader, device=a.device, positions=a.positions,
                          min_n_over_C=a.min_n_over_C, seed=a.seed, progress=True)
        s = res.summary()
        print(f"{a.model}: {s['L']} dense conv layers pass the gate; median k*(0.95)/r_max {s['median_level']:.3f}; "
              f"rho(depth) {s['rho_depth']:+.2f}; max k*(0.999)/r_max {s['rmax_check_max']:.3f}"
              + ("" if s["rmax_check_ok"] else "  ** exceeds 1: r_max is wrong for some layer **"))
    else:
        res = api.decompose(model, loader, device=a.device, positions=a.positions,
                            rank_tol=a.rank_tol, seed=a.seed)
        c = res.check()
        print(f"{a.model}: {len(res.table)} dense conv layers; Ostrowski {c['ostrowski'][0]}/{c['ostrowski'][1]}; "
              f"kappa median {c['kappa_median']:.1f}")
    res.to_csv(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `cd code && ~/.venvs/effwidth/bin/python -m pytest tests -q`
Expected: all pass. Also check the console script resolves: `~/.venvs/effwidth/bin/layerspec --help` prints the two subcommands.

- [ ] **Step 6: Commit**

```bash
git add code/layerspec/api.py code/layerspec/cli.py code/layerspec/__init__.py code/tests/test_api.py
git commit -m "layerspec: load_model/image_loader conveniences and the layerspec CLI"
```

---

### Task 7: Documentation, build check, paper sentence

**Files:**
- Modify: `README.md` (top-level), `code/README.md`
- Modify: `paper/main.tex` Reproducibility section
- Create: `code/CHANGELOG.md`

- [ ] **Step 1: Add the usage section to both READMEs**

Insert after the "Reproducing the numbers" section of `README.md` (and as the first section after the title of `code/README.md`):

````markdown
## Use it on your own model

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

Or from the shell: `layerspec profile --model resnet50 --data /path/to/images --out resnet50.csv`.

What the numbers mean: `k*(tau)/r_max` is the number of principal directions of a
convolution's output covariance (before the nonlinearity) carrying a fraction
`tau` of its variance, divided by the rank the layer can attain
(`groups * min(C_in/groups * kh * kw, C_out/groups)`). Rows with `ok == False`
have fewer than 50 samples per channel and are not reportable. Depthwise
convolutions are flagged and left out of `dense()`. A `k*(0.999)/r_max` above 1
means `r_max` is wrong for that layer type; please report it.
````

- [ ] **Step 2: Fix the stale findings paragraph in `README.md`**

In the "Findings in one paragraph" section replace the sentence beginning `With the protocol fixed, what training changes depends on the block type:` through `share no profile.` with:

```
With the protocol fixed, trained networks largely keep the layer ordering the
architecture gave them and gain a level, and the profile is in place before
accuracy is; a pre-registered controlled experiment on CIFAR-10 shows the block
type does not change this, so the one family whose public checkpoints reorder
their layers, ImageNet ResNet-50, owes it to its training regime.
```

Also change the title quoted at the top of `README.md` to match `paper/main.tex`'s current `\title{}` (copy it verbatim).

- [ ] **Step 3: Paper sentence**

In `paper/main.tex`, in the `\section*{Reproducibility}` block, after `evidence that the analysis was fixed before the ImageNet data were seen.` add:

```latex
The measurement itself is packaged: \texttt{pip install layerspec} gives
\texttt{layerspec.profile(model, loader)} and \texttt{layerspec.decompose(model,
loader)}, which return the per-layer tables of Sections~\ref{sec:method}
and~\ref{sec:decompose} for any PyTorch network with convolutional layers.
```

Check the label `sec:method` exists (`grep -n 'label{sec:method}' paper/main.tex`); if the Method section uses a different label, use that one.

- [ ] **Step 4: CHANGELOG and build**

Create `code/CHANGELOG.md`:

```markdown
# Changelog

## 0.2.0 (2026-09-09)
- Public API: `layerspec.profile()`, `layerspec.decompose()`, `Profile`, `Decomposition`,
  `load_model()`, `image_loader()`, `select_device()`.
- `layerspec` console script with `profile` and `decompose` subcommands.
- The decomposition core moved from `scripts/decompose_check.py` into `layerspec.decompose`;
  the script is now a wrapper and reproduces the paper's ResNet-50 table row for row.
- Packaging via `pyproject.toml`; torchvision/timm are optional (`[models]`).

## 0.1.0
- Measurement code as used for the paper's rounds 1-3.
```

Build check (no upload):
```bash
cd code && ~/.venvs/effwidth/bin/python -m pip install -q build && ~/.venvs/effwidth/bin/python -m build --outdir /tmp/ls_dist 2>&1 | tail -2 && ls /tmp/ls_dist
```
Expected: a `layerspec-0.2.0.tar.gz` and a `layerspec-0.2.0-py3-none-any.whl`. Then install the wheel into a throwaway venv and import it:
```bash
python3 -m venv /tmp/ls_venv && /tmp/ls_venv/bin/pip install -q /tmp/ls_dist/layerspec-0.2.0-py3-none-any.whl && /tmp/ls_venv/bin/python -c "import layerspec; print(layerspec.__version__)"
```
Expected: `0.2.0` (this pulls torch into the throwaway venv; it is slow but proves the dependency list is right). Remove `/tmp/ls_venv` afterwards.

- [ ] **Step 5: Compile the paper and commit**

```bash
cd paper && make tectonic 2>&1 | grep -i "^error\|undefined"; cd ..
git add README.md code/README.md code/CHANGELOG.md paper/main.tex
git commit -m "layerspec 0.2: usage docs, changelog, README findings paragraph updated, paper Reproducibility mentions the package"
```

Expected: tectonic prints no errors; the commit lands.

- [ ] **Step 6: Update the submission bar**

In `notes/submission-bar-2026-09-07.md` change the I1 row's status from `未做` to `✔ 09-09：layerspec 0.2（profile／decompose API、CLI、pyproject、測試）；PyPI 上傳待作者註冊帳號後 `python -m twine upload dist/*``. Commit:

```bash
git add notes/submission-bar-2026-09-07.md
git commit -m "Bar: I1 done (PyPI upload pending the author's account)"
git push origin master
```
