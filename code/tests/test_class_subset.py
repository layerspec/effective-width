import numpy as np
from scripts.class_subset import classes_for, label_map, subset_indices, RemapSubset


def test_nested_and_sorted():
    c10, c20, c50, c100 = (classes_for(100, k) for k in (10, 20, 50, 100))
    assert set(c10) < set(c20) < set(c50) < set(c100) == set(range(100))
    assert c10 == sorted(c10) and len(c10) == 10


def test_subset_indices_and_remap():
    rng = np.random.RandomState(1)
    targets = rng.randint(0, 100, size=5000)
    cls = classes_for(100, 10)
    idx = subset_indices(targets, cls, per_class=7, seed=0)
    assert len(idx) == 70 and set(targets[idx]) == set(cls)
    assert list(idx) == sorted(idx)
    idx_all = subset_indices(targets, cls, None)
    assert len(idx_all) == int(np.isin(targets, cls).sum())
    lm = label_map(cls)
    assert sorted(lm.values()) == list(range(10))
    base = [(i, int(t)) for i, t in enumerate(targets)]
    ds = RemapSubset(base, idx, lm)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    assert set(ys) == set(range(10)) and list(xs) == [int(i) for i in idx]


def test_defaults_are_identity():
    assert classes_for(10, None) == list(range(10))
    t = np.arange(10).repeat(3)
    assert len(subset_indices(t, classes_for(10, None), None)) == 30
