"""Width reallocation for any network that can list its Sites.

    from layerspec.realloc import discover_sites, measure, shrink, grow, water_fill

    sites = discover_sites(model)                    # or your own list of Site
    mom   = measure(model, sites, calib_loader, device)
    order, traces = greedy_cssp(mom[s.name].act_cov)  # Proposition A: which channels to keep
    shrink(model, s, order[:m], mom[s.name])          # fold the rest into the consumers
    grow(model, s, k, mom[s.name])                    # Proposition B: function-preserving growth
    widths, mu, _ = water_fill(spectra, cost, budget) # Proposition C: where the channels go
"""
from .allocate import water_fill
from .ops import greedy_cssp, grow, reconstruction, removable_width, set_module, shrink, unmet_directions
from .sites import Consumer, Site, discover_sites, mlp, resnet_internal, sequential_conv
from .stats import Moments, measure

__all__ = ["Site", "Consumer", "discover_sites", "sequential_conv", "resnet_internal", "mlp",
           "Moments", "measure", "greedy_cssp", "removable_width", "reconstruction", "shrink", "grow",
           "unmet_directions", "set_module", "water_fill"]
