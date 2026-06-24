"""Inverse Box-Cox transform. Port of .inv_box_cox in R/pts-internals.R.

Branch convention matches the C++ engine exactly: lambda == 0 -> exp,
lambda == 1 -> identity, otherwise the general (1 + lambda*x)^(1/lambda)
with support-boundary handling for arguments <= 0.
"""
from __future__ import annotations

import numpy as np


def inv_box_cox(x, lam: float):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if lam == 1:
        return x
    if lam == 0:
        return np.exp(x)
    arg = lam * x + 1.0
    out = np.empty_like(x)
    pos = arg > 0
    out[pos] = arg[pos] ** (1.0 / lam)
    # boundary: 0 for lambda > 0 (Y >= 0), +inf for lambda < 0
    out[~pos] = 0.0 if lam > 0 else np.inf
    return out


def inv_box_cox_mean(mu_bc, variance_bc, lam: float):
    """Inverse Box-Cox returning the conditional MEAN, not the median.

    Port of .inv_box_cox_mean in R/pts-internals.R.  inv_box_cox(mu) is the
    conditional median on the original scale; since the inverse transform is
    convex for lambda < 1 the conditional mean E[g^{-1}(z)], z ~ N(mu, var), is
    higher.  Second-order (delta-method) bias correction, matching
    forecast::InvBoxCox(biasadj=True):

        E[g^{-1}(z)] ~= g^{-1}(mu) * (1 + 0.5*var*(1-lambda)/(1+lambda*mu)^2).

    Exact (factor 1) at lambda == 1.  Falls back to no correction at the
    support boundary (1 + lambda*mu <= 0) or any non-finite/negative factor.
    """
    med = inv_box_cox(mu_bc, lam)
    if lam == 1 or np.asarray(med).size == 0:
        return med
    mu = np.asarray(mu_bc, dtype=float)
    v = np.asarray(variance_bc, dtype=float)
    base = 1.0 + lam * mu
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = 1.0 + 0.5 * v * (1.0 - lam) / (base * base)
    corr = np.where(np.isfinite(corr) & (base > 0) & (corr >= 0), corr, 1.0)
    return med * corr
