"""Box-Cox lambda screen via classical decomposition + Guerrero CV.

Port of .pts_guerrero_decomp_lambda (R/pts-internals.R).  Reuses the Python
`smooth` package's msdecompose (smoother="ma") for the trend, then minimises
the coefficient of variation of sigma_b * mu_b^(lambda-1) across
non-overlapping seasonal blocks, over the clipped range [0, 2].
"""
from __future__ import annotations

import numpy as np


def _nansd1(x):
    """Sample sd (ddof=1) over finite values, matching R's sd(., na.rm=TRUE)."""
    v = x[np.isfinite(x)]
    if v.size < 2:
        return np.nan
    return float(np.std(v, ddof=1))


def guerrero_decomp_lambda(y, lags, lower: float = 0.0, upper: float = 2.0) -> float:
    y = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        return 1.0
    m = int(np.atleast_1d(lags)[-1])
    if m < 2:
        return 1.0
    n = y.size
    if n < 2 * m:
        return 1.0

    try:
        from smooth import msdecompose
        decomp = msdecompose(y, lags=[m], type="additive", smoother="ma")
    except Exception:
        return 1.0

    mu_t = np.asarray(decomp["states"][:, 0], dtype=float)  # smoothed level

    R = n // m
    keep = R * m
    block = np.repeat(np.arange(R), m)
    mu_blk = mu_t[:keep]
    dev_blk = (y[:keep] - mu_t[:keep])
    # nan-aware to match R's tapply(..., na.rm = TRUE): the centred-MA level
    # is NaN over the boundary half-blocks, but those blocks still
    # contribute finite stats from their non-NaN half.
    with np.errstate(invalid="ignore"):
        mu_b = np.array([np.nanmean(mu_blk[block == i]) for i in range(R)])
        sd_b = np.array([_nansd1(dev_blk[block == i]) for i in range(R)])

    ok = np.isfinite(mu_b) & np.isfinite(sd_b) & (mu_b > 0) & (sd_b > 0)
    mu_b, sd_b = mu_b[ok], sd_b[ok]
    if mu_b.size < 2:
        return 1.0
    if lower >= upper:
        return float(lower)

    def cv(lam):
        r = sd_b * mu_b ** (lam - 1.0)
        if not np.all(np.isfinite(r)):
            return np.inf
        return r.std(ddof=1) / r.mean()

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        cv, bounds=(lower, upper), method="bounded", options={"xatol": 1e-4}
    )
    if not res.success or not np.isfinite(res.fun):
        return 1.0
    return float(res.x)
