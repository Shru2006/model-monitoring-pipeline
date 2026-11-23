"""Checks for the metrics, the simulator and the monitor's no-peeking rule.

Run:  pytest -q
"""
import numpy as np
import pytest

from src.config import LABEL_LAG_MONTHS
from src.metrics import psi, top_decile_capture
from src.monitor import run
from src.simulate import shock, simulate


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20_000), rng.normal(0, 1, 20_000)
    assert psi(a, b) < 0.01
    assert psi(a, rng.normal(1, 1, 20_000)) > 0.2


def test_psi_handles_discrete_features():
    rng = np.random.default_rng(1)
    assert psi(rng.binomial(1, 0.06, 5000), rng.binomial(1, 0.06, 5000)) < 0.02


def test_top_decile_capture():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert top_decile_capture(y, np.arange(10, 0, -1)) == pytest.approx(0.5)


def test_shock_ramps_from_month_8():
    assert shock(7) == 0 and shock(14) == 1 and 0 < shock(10) < 1


def test_simulated_drift_moves_the_expected_way():
    df = simulate(per_month=2000, seed=3)
    early, late = df[df.month <= 6], df[df.month >= 14]
    assert late.savings_growth_6m.mean() < early.savings_growth_6m.mean()
    assert late.calculator_uses_90d.mean() > early.calculator_uses_90d.mean()


@pytest.fixture(scope="module")
def result():
    return run(simulate(per_month=2500, seed=7))


def test_drift_alert_fires_after_the_shock(result):
    assert result["drift_alert_month"] is not None and result["drift_alert_month"] >= 8


def test_retraining_never_uses_labels_that_were_not_yet_known(result):
    for policy in result["policies"].values():
        for r in policy["retrains"]:
            assert r["trained_on"][1] + LABEL_LAG_MONTHS <= r["month"]
