"""Tests for forecast evaluation metrics.

Covers the QLIKE definition, OOS R² semantics, and Diebold–Mariano
numerical stability and symmetry under the two-sided null.
"""

from __future__ import annotations

import numpy as np
import pytest
from rvforecast.evaluation.metrics import diebold_mariano, qlike, r2_oos


def test_qlike_zero_when_pred_equals_truth():
    truth = np.array([0.04, 0.09, 0.16])
    assert qlike(truth, truth.copy()) == pytest.approx(0.0)


def test_qlike_positive_for_biased_pred():
    truth = np.array([0.04, 0.09, 0.16])
    pred = truth * 2.0
    assert qlike(truth, pred) > 0.0


def test_qlike_pinned_against_hand_calculation():
    """Pin QLIKE against a hand-calculated value.

    With ``x = sigma2_true / sigma2_pred``, QLIKE per-row is ``x − log(x) − 1``.
    For truth=[0.04, 0.09], pred=[0.05, 0.08]:
        x = [0.8, 1.125]
        per-row = [0.8 − log(0.8) − 1, 1.125 − log(1.125) − 1]
                ≈ [0.0231436, 0.0072170]
        mean    ≈ 0.0151803
    A swapped numerator/denominator, missing −1, or missing log would all
    fail this assertion; the existing zero-at-equality and sign tests would
    not.
    """
    truth = np.array([0.04, 0.09])
    pred = np.array([0.05, 0.08])
    assert qlike(truth, pred) == pytest.approx(0.0151803, abs=1e-6)


def test_dm_positive_when_b_has_lower_loss():
    """A uniformly-better model B yields a positive DM statistic.

    Pins the documented sign convention: ``d = errors_a − errors_b``, so a
    smaller ``errors_b`` produces ``d > 0`` on average and thus DM > 0. A
    sign flip in either the docstring or the formula would fail this test.
    """
    rng = np.random.default_rng(1)
    n = 500
    err_a = 1.0 + rng.normal(0.0, 0.1, size=n)
    err_b = rng.normal(0.0, 0.1, size=n)
    dm, p = diebold_mariano(err_a, err_b)
    assert dm > 0.0
    assert p < 0.01


def test_dm_negative_when_a_has_lower_loss():
    """Mirror of the above: a uniformly-better A yields negative DM."""
    rng = np.random.default_rng(2)
    n = 500
    err_a = rng.normal(0.0, 0.1, size=n)
    err_b = 1.0 + rng.normal(0.0, 0.1, size=n)
    dm, p = diebold_mariano(err_a, err_b)
    assert dm < 0.0
    assert p < 0.01


def test_r2_oos_is_one_when_model_is_perfect():
    truth = np.linspace(0.1, 1.0, 50)
    base = np.zeros_like(truth)
    assert r2_oos(truth, truth.copy(), base) == pytest.approx(1.0)


def test_dm_symmetric_under_two_sided_null():
    rng = np.random.default_rng(0)
    err_a = rng.normal(size=500) ** 2
    err_b = rng.normal(size=500) ** 2
    dm_ab, p_ab = diebold_mariano(err_a, err_b)
    dm_ba, p_ba = diebold_mariano(err_b, err_a)
    assert dm_ab == pytest.approx(-dm_ba)
    assert p_ab == pytest.approx(p_ba)


def test_dm_returns_nan_on_zero_variance_loss_diff():
    """Identical loss series yield a degenerate variance; result must be NaN.

    The previous implementation floored the variance at ``1e-12`` and
    produced a colossal ``dm`` and a meaningless near-zero p-value. We
    now return ``(nan, nan)`` so callers see the degeneracy explicitly.
    """
    err_a = np.full(1000, 10.0)
    err_b = np.full(1000, 0.0)
    dm, p = diebold_mariano(err_a, err_b)
    assert np.isnan(dm)
    assert np.isnan(p)


def test_dm_p_value_stable_on_large_finite_dm():
    """``norm.sf`` must keep the p-value strictly positive for large |dm|.

    Construct a non-degenerate loss-difference series with separation in
    the 10-30 sigma range. There ``1 - norm.cdf(dm)`` rounds to 0 in
    double precision, while ``norm.sf(dm)`` returns a tiny but positive
    value. (Past ~37 sigma even ``sf`` underflows; the regime tested here
    is the one the implementation actually has to handle.)
    """
    rng = np.random.default_rng(0)
    n = 1000
    err_a = 1.0 + rng.normal(0.0, 1.0, size=n)
    err_b = rng.normal(0.0, 1.0, size=n)
    dm, p = diebold_mariano(err_a, err_b)
    assert np.isfinite(dm)
    assert 10.0 < abs(dm) < 35.0
    assert 0.0 < p < 1e-20
