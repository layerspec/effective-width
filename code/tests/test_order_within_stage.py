import numpy as np, pandas as pd
from scripts.order_within_stage import pooled_rank_rho, role_of, per_cell

def _prof(vals, stages, roles):
    return pd.DataFrame({"v": vals, "stage": stages, "role": roles, "C": 64}, index=[f"l{i}" for i in range(len(vals))])

def test_pooled_rank_invariant_to_level_shift_and_denominator():
    rng = np.random.default_rng(0)
    stages = ["s1"] * 6 + ["s2"] * 6; roles = ["3x3"] * 12
    a = _prof(rng.random(12), stages, roles)
    b = _prof(a.v.values + np.where(np.array(stages) == "s2", 5.0, 0.0), stages, roles)  # level shift of stage 2
    c = _prof(a.v.values * np.where(np.array(stages) == "s2", 4.0, 1.0), stages, roles)  # another denominator in stage 2
    assert abs(pooled_rank_rho(a, b)[0] - 1.0) < 1e-12
    assert abs(pooled_rank_rho(a, c)[0] - 1.0) < 1e-12

def test_pooled_rank_detects_within_stage_reversal():
    stages = ["s1"] * 6 + ["s2"] * 6; roles = ["3x3"] * 12
    a = _prof(np.arange(12, dtype=float), stages, roles)
    v = a.v.values.copy(); v[6:] = v[6:][::-1]
    b = _prof(v, stages, roles)
    rho, n = pooled_rank_rho(a, b)
    assert n == 12 and rho < 0.1
    cells = {(c["stage"], c["role"]): c["rho"] for c in per_cell(a, b) if c["stage"] != "all"}
    assert cells[("s1", "3x3")] > 0.99 and cells[("s2", "3x3")] < -0.99

def test_roles():
    assert role_of("resnet50", "layer1.0.conv2") == "3x3" and role_of("resnet50", "layer1.0.conv3") == "1x1"
    assert role_of("resnet18", "layer2.1.conv1") == "3x3" and role_of("resnet18", "layer2.0.downsample.0") == "1x1"
    assert role_of("vgg16", "features.27") == "3x3" and role_of("mobilenetv2", "features.3.conv.2") == "1x1"
