"""Forecasting + prediction intervals. Port of forecast.pts (R/methods.R).

Intervals are computed by endpoint-transforming the BC-scale +/- z*se bounds
through the inverse Box-Cox (with the lambda<0 truncated-distribution
renormalisation), matching R exactly.  No greybox dependency: R's forecast
path uses base pnorm/qnorm, which scipy.stats.norm reproduces.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .boxcox import inv_box_cox, inv_box_cox_mean


class ForecastResult:
    def __init__(self, mean, lower, upper, variance, level, interval, side,
                 cumulative, scenarios=None):
        self.mean = mean
        self.lower = lower
        self.upper = upper
        self.variance = variance
        self.level = level
        self.interval = interval
        self.side = side
        self.cumulative = cumulative
        self.scenarios = scenarios

    def __repr__(self):
        lv = np.atleast_1d(self.level)
        return (
            f"ForecastResult(h={np.atleast_1d(self.mean).size}, "
            f"interval={self.interval!r}, level={list(lv)}, side={self.side!r})"
        )


def _tail_probs(level, side):
    level = np.atleast_1d(np.asarray(level, dtype=float))
    nL = level.size
    if side == "both":
        return (1 - level) / 2, (1 + level) / 2
    if side == "upper":
        return np.zeros(nL), level
    if side == "lower":
        return 1 - level, np.ones(nL)
    raise ValueError(side)


def _bands(fn, probs, h):
    """(h x nLevels) matrix from evaluating fn(prob) per level."""
    probs = np.atleast_1d(probs)
    out = np.empty((h, probs.size), dtype=float)
    for j, p in enumerate(probs):
        out[:, j] = np.asarray(fn(p), dtype=float)
    return out


def forecast(model, h, X=None, interval="prediction", level=0.95, side="both",
             cumulative=False, nsim=10000, seed=0, scenarios=False):
    if h < 1:
        raise ValueError("h must be a positive integer.")
    level = np.atleast_1d(np.asarray(level, dtype=float))
    if np.any(level <= 0) or np.any(level >= 1):
        raise ValueError("level must be in (0, 1).")
    qLow, qUp = _tail_probs(level, side)

    eng = model._forecast_engine(h, X=X)
    yfor_bc = np.asarray(eng["yFor"], dtype=float)
    yforv = np.asarray(eng["yForV"], dtype=float)
    lam = model._lambda
    # Point forecast = conditional MEAN (bias-corrected back-transform); the
    # interval quantiles below stay median-style (exact quantiles, no bias adj).
    mean_out = inv_box_cox_mean(yfor_bc, yforv, lam)

    sigma2_bc = float(model._scale) ** 2
    yforv_conf = np.maximum(0.0, yforv - sigma2_bc)

    def bc_quant(p, var):
        se = np.sqrt(var)
        if lam < 0:
            x_max = -1.0 / lam
            p_valid = norm.cdf(x_max, loc=yfor_bc, scale=se)
            x_q = norm.ppf(p * p_valid, loc=yfor_bc, scale=se)
            return inv_box_cox(x_q, lam)
        z = norm.ppf(p)
        if np.isfinite(z):
            return inv_box_cox(yfor_bc + z * se, lam)
        if z == -np.inf:
            return inv_box_cox(np.full(h, -np.inf), lam)
        return np.full(h, np.inf)

    paths_cache = {}

    def draw_paths():
        if "p" not in paths_cache:
            paths_cache["p"] = model._forecast_paths(h, nsim=nsim, seed=seed)
        return paths_cache["p"]

    scen = None
    if interval == "none":
        lower = _bands(lambda p: mean_out, qLow, h)
        upper = _bands(lambda p: mean_out, qUp, h)
        variance = yforv
    elif interval in ("prediction", "confidence"):
        variance = yforv if interval == "prediction" else yforv_conf
        lower = _bands(lambda p: bc_quant(p, variance), qLow, h)
        upper = _bands(lambda p: bc_quant(p, variance), qUp, h)
    elif interval == "simulated":
        paths = draw_paths()  # (h x nsim), original scale

        def q_paths(p):
            if p <= 0:
                return np.full(h, -np.inf)
            if p >= 1:
                return np.full(h, np.inf)
            return np.quantile(paths, p, axis=1)

        lower = _bands(q_paths, qLow, h)
        upper = _bands(q_paths, qUp, h)
        variance = np.var(paths, axis=1, ddof=1)
        if scenarios:
            scen = paths
    else:
        raise ValueError(f"unknown interval {interval!r}")

    if cumulative:
        mean_scalar = float(np.nansum(mean_out))
        if interval == "none":
            mean_out = mean_scalar
            lower = np.full((1, level.size), mean_scalar)
            upper = np.full((1, level.size), mean_scalar)
            variance = 0.0
        else:
            totals = draw_paths().sum(axis=0)

            def q_tot(p):
                if p <= 0:
                    return -np.inf
                if p >= 1:
                    return np.inf
                return float(np.quantile(totals, p))

            mean_out = mean_scalar
            lower = np.array([[q_tot(p) for p in qLow]])
            upper = np.array([[q_tot(p) for p in qUp]])
            variance = float(np.var(totals, ddof=1))

    return ForecastResult(mean_out, lower, upper, variance,
                          level if level.size > 1 else float(level[0]),
                          interval, side, cumulative, scen)
