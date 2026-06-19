"""Functionality / invariant tests for the muse Python package.

These check that the package *works* (shapes, finiteness, orderings,
round-trips) without comparing against R -- so they run in CI with no R
dependency and no reference dumps.  The R<->Python numeric parity lives in
the separate test_*_parity.py scripts (not run in CI)."""
import importlib.util

import numpy as np
import pytest

from muse import PTS

LAGS = 12
# Auto-lambda (power "Z") needs smooth.msdecompose; skip those paths when
# smooth (not on PyPI) is unavailable, so engine-only functionality still
# runs in CI.  Power-fixed + auto trend/seasonal ("1ZZ") needs no smooth.
HAS_SMOOTH = importlib.util.find_spec("smooth") is not None
needs_smooth = pytest.mark.skipif(not HAS_SMOOTH, reason="smooth not installed")


def _series(n=120, seed=0):
    """Deterministic positive seasonal series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 50 + 1.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / LAGS) + 8 * np.cos(2 * np.pi * t / LAGS)
    noise = rng.normal(0, 5, n)
    return np.maximum(1.0, np.round(trend + seasonal + noise))


Y = _series()

SPECS = ["1NN", "1LN", "1LT", "1DT", "1GT", "1ND", "0LT", "0.5LT", "2LT",
         "1ZN", "1ZZ"]


@pytest.mark.parametrize("spec", SPECS)
def test_fit_invariants(spec):
    m = PTS(model=spec, lags=LAGS, h=6, holdout=True).fit(Y)
    n_in = len(Y) - 6
    assert np.asarray(m.fitted).shape[0] == n_in
    assert np.asarray(m.residuals).shape[0] == n_in
    assert m.nobs == n_in
    assert np.all(np.isfinite(np.asarray(m.fitted)))
    assert np.all(np.isfinite(np.asarray(m.residuals)))
    assert np.isfinite(m.log_lik)
    assert np.isfinite(m.sigma) and m.sigma > 0
    assert m.n_param >= 1
    # cached forecast present and finite
    fc = m.predict(6, interval="none")
    assert np.asarray(fc.mean).shape[0] == 6
    assert np.all(np.isfinite(np.asarray(fc.mean)))


def test_information_criteria_consistency():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    for ic in (m.aic, m.bic, m.aicc, m.bicc):
        assert np.isfinite(ic)
    assert m.aicc >= m.aic - 1e-8
    assert m.bicc >= m.bic - 1e-8
    assert abs(m.aic - (-2 * m.log_lik + 2 * m.n_param)) < 1e-6


def test_prediction_intervals_ordered_and_nested():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    fc = m.predict(LAGS, interval="prediction", level=[0.8, 0.95])
    lo, up, mean = np.asarray(fc.lower), np.asarray(fc.upper), np.asarray(fc.mean)
    assert np.all(lo[:, 0] <= mean + 1e-6)
    assert np.all(up[:, 0] >= mean - 1e-6)
    assert np.all(lo[:, 1] <= lo[:, 0] + 1e-6)   # 95% wider than 80%
    assert np.all(up[:, 1] >= up[:, 0] - 1e-6)
    assert np.all(lo >= -1e-6)                   # positive series, lambda>=0


def test_confidence_narrower_than_prediction():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    fp = m.predict(LAGS, interval="prediction", level=0.95)
    fcid = m.predict(LAGS, interval="confidence", level=0.95)
    wp = np.asarray(fp.upper).ravel() - np.asarray(fp.lower).ravel()
    wc = np.asarray(fcid.upper).ravel() - np.asarray(fcid.lower).ravel()
    assert np.all(wc <= wp + 1e-6)


def test_forecast_sides_and_cumulative():
    m = PTS(model="1LT", lags=LAGS).fit(Y)
    assert np.asarray(m.predict(6, side="upper").mean).shape[0] == 6
    assert np.asarray(m.predict(6, side="lower").mean).shape[0] == 6
    fcum = m.predict(6, interval="prediction", cumulative=True)
    assert np.isscalar(fcum.mean) or np.asarray(fcum.mean).size == 1


@pytest.mark.parametrize("ar,ma", [(1, 0), (0, 1), (1, 1)])
def test_arma_orders_reported(ar, ma):
    m = PTS(model="1LT", lags=LAGS, orders={"ar": ar, "ma": ma}).fit(Y)
    assert m.orders["ar"] == ar
    assert m.orders["ma"] == ma
    assert np.all(np.isfinite(np.asarray(m.residuals)))


def test_orders_select_within_caps():
    m = PTS(model="1LT", lags=LAGS, ic="AICc",
            orders={"ar": 2, "ma": 2, "select": True}).fit(Y)
    assert m.orders["ar"] <= 2 and m.orders["ma"] <= 2
    assert m.orders["select"] is True


def test_diagnostics():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    n = m.nobs
    assert np.asarray(m.rstandard()).shape[0] == n
    assert np.asarray(m.rstudent()).shape[0] == n
    assert np.asarray(m.point_lik()).shape[0] == n
    assert np.all(np.isfinite(np.asarray(m.rstandard())))
    assert abs(np.std(np.asarray(m.rstandard())) - 1) < 0.5


def test_simulate_reproducible():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    s1 = np.asarray(m.simulate(nsim=30, seed=7))
    s2 = np.asarray(m.simulate(nsim=30, seed=7))
    assert s1.shape == (m.nobs, 30)
    assert np.all(np.isfinite(s1))
    assert np.array_equal(s1, s2)


def test_confint_shapes():
    m = PTS(model="1LT", lags=LAGS).fit(Y)
    ci = m.confint(level=0.9)
    assert len(ci["lower"]) == m.coef_values.size
    assert len(ci["upper"]) == m.coef_values.size


def test_summary_structure():
    m = PTS(model="1ZZ", lags=LAGS).fit(Y)
    s = m.summary()
    c, p = s["coefficients"], s["proportions"]
    assert len(c["names"]) == c["estimate"].size
    # proportions sum to 1
    assert abs(float(np.sum(p["proportion"])) - 1.0) < 1e-8


def test_update_matches_fresh_fit():
    m = PTS(model="1LT", lags=LAGS, h=6, holdout=True).fit(Y)
    m2 = m.update(h=12)
    fresh = PTS(model="1LT", lags=LAGS, h=12, holdout=True).fit(Y)
    assert np.allclose(m2.coef_values, fresh.coef_values)


@pytest.mark.parametrize("spec", ["0LT", "0.5LT", "1LT"])
def test_boxcox_positivity(spec):
    m = PTS(model=spec, lags=LAGS).fit(Y)
    assert np.all(np.asarray(m.fitted) > 0)
    assert np.all(np.asarray(m.predict(LAGS, interval="none").mean) > 0)


def test_outliers_use_detects_spike():
    y = Y.copy()
    y[60] *= 2.5
    m = PTS(model="1LT", lags=LAGS, outliers="use", level=0.99).fit(y)
    times = [d["time"] for d in m.outliers_detected]
    assert len(m.outliers_detected) >= 1
    assert any(abs(t - 61) <= 1 for t in times)  # 1-based, near the spike


def test_pandas_series_input_infers_lags():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2000-01-01", periods=len(Y), freq="MS")
    s = pd.Series(Y, index=idx)
    m = PTS(model="1ZZ", h=12, holdout=True).fit(s)   # no lags passed
    assert m._lags == 12
    assert isinstance(m.fitted, pd.Series)
    assert isinstance(m.actuals, pd.Series)


def test_invalid_inputs_raise():
    with pytest.raises(NotImplementedError):
        PTS(model="1LT", outliers="select")
    with pytest.raises(ValueError):
        PTS(model="1LT", level=1.5)
    with pytest.raises(ValueError):
        PTS(model="1LT", lags=None).fit(np.asarray(Y))  # array, no lags


@needs_smooth
@pytest.mark.parametrize("spec", ["ZNN", "ZLT", "ZZZ"])
def test_auto_lambda_screen(spec):
    """Power 'Z' triggers the decomposition+Guerrero screen (needs smooth)."""
    m = PTS(model=spec, lags=LAGS, h=6, holdout=True).fit(Y)
    assert 0.0 <= m.lambda_ <= 2.0          # screen clips lambda to [0, 2]
    assert np.isfinite(m.log_lik)
    assert np.all(np.isfinite(np.asarray(m.predict(6, interval="none").mean)))


@needs_smooth
def test_auto_lambda_plus_select():
    m = PTS(model="ZZZ", lags=LAGS, ic="AICc",
            orders={"ar": 2, "ma": 2, "select": True}).fit(Y)
    assert 0.0 <= m.lambda_ <= 2.0
    assert m.orders["ar"] <= 2 and m.orders["ma"] <= 2
