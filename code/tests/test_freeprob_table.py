"""scripts/freeprob_table.py: supplement table of the layer-local law check from results/freeprob/*.csv."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.freeprob_table import render   # noqa: E402


def test_render_lists_every_layer_with_law_and_measurement(tmp_path):
    df = pd.DataFrame([dict(layer="features.0", depth_index=0, C=64, d=27, r_max=27, measured_95=3, mc_95=3.5, mc_sd_95=0.5,
                            silverstein_95=3, measured_999=12, mc_999=12.8, mc_sd_999=0.4, silverstein_999=13),
                       dict(layer="features.3", depth_index=1, C=64, d=576, r_max=64, measured_95=12, mc_95=9.5, mc_sd_95=1.5,
                            silverstein_95=10, measured_999=58, mc_999=56.2, mc_sd_999=0.8, silverstein_999=56)])
    tex = render({"VGG-16 (BN)": df})
    assert tex.count("\\\\") >= 3 and "features.3" in tex and "9.5" in tex and "$\\pm$" in tex and "VGG-16" in tex


def test_summary_table_one_row_per_family_and_tau(tmp_path):
    from scripts.freeprob_table import render_summary
    df = pd.DataFrame([dict(layer="a", depth_index=0, C=64, d=27, r_max=27, measured_95=3, mc_95=3.5, mc_sd_95=0.5, silverstein_95=3,
                            measured_999=12, mc_999=12.8, mc_sd_999=0.4, silverstein_999=13),
                       dict(layer="b", depth_index=1, C=64, d=576, r_max=64, measured_95=12, mc_95=9.5, mc_sd_95=1.5, silverstein_95=10,
                            measured_999=58, mc_999=56.2, mc_sd_999=0.8, silverstein_999=56)])
    tex = render_summary({"VGG-16 (BN)": df, "ResNet-18": df})
    assert tex.count("\\\\") >= 4 and "VGG-16 (BN)" in tex and "ResNet-18" in tex and "0.95" in tex and "0.999" in tex
