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
