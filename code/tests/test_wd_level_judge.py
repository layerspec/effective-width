"""Tests for scripts/wd_level_judge.py (plan 9.30 B)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.wd_level_judge import verdict      # noqa: E402


def _R(levels, accs):
    return {(lam, s): dict(level=lv, acc=ac) for (lam, lv), ac in zip(levels.items(), accs) for s in (0, 1)}


def test_holds_when_eligible_arms_decrease():
    v, elig = verdict(_R({"5e-5": 0.4, "5e-4": 0.08, "1e-3": 0.07, "5e-3": 0.05}, [93.0, 94.0, 93.5, 90.0]))
    assert v is True and elig[0] == ["5e-5", "5e-4", "1e-3"]


def test_fails_when_a_new_arm_is_not_lower():
    v, _ = verdict(_R({"5e-5": 0.4, "5e-4": 0.08, "1e-3": 0.09}, [93.0, 94.0, 93.5]))
    assert v is False


def test_not_judged_without_an_eligible_new_arm():
    v, _ = verdict(_R({"5e-5": 0.4, "5e-4": 0.08, "1e-3": 0.07, "2e-3": 0.06}, [93.0, 94.0, 91.0, 90.0]))
    assert v is None
