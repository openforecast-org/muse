"""PTS-then-ARMA order selection. Port of .pts_select_pts_arma /
.pts_select_arma_at_lag (R/pts-translate.R).

Two sequential passes: rank PTS structural shapes by IC (Pass 1), then run a
cascaded ARMA grid on the winning structural's residuals (Pass 2).  ARMA
candidates are scored by the shared engine via _musecore.ucomp_arma.
"""
from __future__ import annotations

import math

import numpy as np

from .. import _musecore  # type: ignore[attr-defined]

_TREND_ORDER = ["N", "L", "G", "D"]
_SEAS_ORDER = ["N", "D", "T"]


def _ic_from_ll(ll, k, n_obs, ic):
    if not math.isfinite(ll) or k < 1:
        return math.inf
    denom = max(1, n_obs - k - 1)
    if ic == "AIC":
        return -2 * ll + 2 * k
    if ic == "BIC":
        return -2 * ll + math.log(n_obs) * k
    if ic == "AICc":
        return -2 * ll + 2 * k + (2 * k * (k + 1)) / denom
    if ic == "BICc":
        return -2 * ll + math.log(n_obs) * k + (math.log(n_obs) * k * (k + 1)) / denom
    raise ValueError(ic)


def _ivec(x):
    return np.asarray(np.atleast_1d(x), dtype=np.int64)


def select_arma_at_lag(residuals, ar_max, ma_max, lag, ic):
    """One cascade rung: score a (p, q) grid at a single lag, return the
    IC-winning (ar, ma) and that fit's residuals for the next rung."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0, 0, residuals

    best_ic = math.inf
    best = (0, 0, residuals)
    for p in range(int(ar_max) + 1):
        for q in range(int(ma_max) + 1):
            if lag == 1:
                res = _musecore.ucomp_arma(r, _ivec(p), _ivec(q), _ivec(1))
            else:
                res = _musecore.ucomp_arma(
                    r, _ivec([0, p]), _ivec([0, q]), _ivec([1, lag])
                )
            if not res.get("succeed", False):
                continue
            val = res.get(ic, math.inf)
            if val is None or not math.isfinite(val):
                continue
            if val < best_ic:
                best_ic = val
                best = (p, q, np.asarray(res["residuals"], dtype=float))
    return best


def select_pts_arma(
    y, model_template, lags, ar_max, ma_max, arma_lags, ic, fit_structural
):
    """Pass 1 (structural shapes) + Pass 2 (cascaded ARMA on the winner).

    `fit_structural(model_spec)` must return a dict with keys
    logLik, residuals, n_p (len of coef), lambda, lambdaEstimated,
    n_initial (engine ns(0)+ns(1)+ns(2)).
    Returns a dict: model_spec, lambda, trend, seasonal, ar, ma, lags.
    """
    n = len(model_template)
    lambda_str = model_template[: n - 2]
    user_trend = model_template[n - 2].upper()
    user_seas = model_template[n - 1].upper()
    trend_cands = _TREND_ORDER if user_trend == "Z" else [user_trend]
    seas_cands = _SEAS_ORDER if user_seas == "Z" else [user_seas]
    trend_cands = [t for t in _TREND_ORDER if t in trend_cands]
    seas_cands = [s for s in _SEAS_ORDER if s in seas_cands]

    n_obs = len(np.asarray(y, dtype=float))

    # Pass 1 -- rank structural shapes.
    best = {"struct_ic": math.inf}
    for tr in trend_cands:
        for se in seas_cands:
            spec = f"{lambda_str}{tr}{se}"
            try:
                st = fit_structural(spec)
            except Exception:
                continue
            if st is None:
                continue
            # k = optimised params + lambda + estimated diffuse structural
            # initials (engine ns(0)+ns(1)+ns(2)).  n_initial already includes
            # the G/td drift (= initial slope), so no separate "tr == 'G'".
            k = st["n_p"] + int(bool(st["lambdaEstimated"])) + st["n_initial"]
            sic = _ic_from_ll(st["logLik"], k, n_obs, ic)
            if not math.isfinite(sic):
                continue
            if sic < best["struct_ic"]:
                best = {
                    "struct_ic": sic, "model_spec": spec, "lambda": st["lambda"],
                    "trend": tr, "seasonal": se, "resid": st["residuals"],
                }
    if not math.isfinite(best["struct_ic"]):
        raise RuntimeError("PTS+ARMA selection: no finite structural candidate.")

    # Pass 2 -- cascaded ARMA, highest seasonal lag first, non-seasonal last.
    ar_local = list(_ivec(ar_max))
    ma_local = list(_ivec(ma_max))
    arma_lags = list(_ivec(arma_lags))
    if best["trend"] == "D":
        ar_local = [0] * len(ar_local)  # (damped, AR>0) exclusion

    order = sorted(range(len(arma_lags)), key=lambda i: arma_lags[i], reverse=True)
    ar_chosen = [0] * len(arma_lags)
    ma_chosen = [0] * len(arma_lags)
    resid_chain = np.asarray(best["resid"], dtype=float)
    for k in order:
        p, q, resid_chain = select_arma_at_lag(
            resid_chain, ar_local[k], ma_local[k], arma_lags[k], ic
        )
        ar_chosen[k] = p
        ma_chosen[k] = q

    # Drop trailing zero blocks (keep the non-seasonal lag-1 block).
    keep = [
        i
        for i in range(len(arma_lags))
        if ar_chosen[i] > 0 or ma_chosen[i] > 0 or arma_lags[i] == 1
    ]
    if not keep:
        out_ar, out_ma, out_lags = 0, 0, [1]
    elif len(keep) == 1 and arma_lags[keep[0]] == 1:
        out_ar, out_ma, out_lags = ar_chosen[keep[0]], ma_chosen[keep[0]], [1]
    else:
        out_ar = [ar_chosen[i] for i in keep]
        out_ma = [ma_chosen[i] for i in keep]
        out_lags = [arma_lags[i] for i in keep]

    return {
        "model_spec": best["model_spec"], "lambda": best["lambda"],
        "trend": best["trend"], "seasonal": best["seasonal"],
        "ar": out_ar, "ma": out_ma, "lags": out_lags,
    }
