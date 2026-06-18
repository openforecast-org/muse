"""PTS - Power/Trend/Seasonal state-space model (Python front-end).

Phase-1 estimation path: a scikit-learn-style estimator class mirroring
smooth.ADAM.  Spec goes in the constructor, data into fit(y), results are
exposed as properties.  Wraps the shared C++ engine via muse._musecore.

Scope of this phase: estimation + in-sample fit + information criteria for
fixed-lambda specs (and engine-side trend/seasonal "Z" selection, which the
engine resolves).  The R-side auto-lambda screen, ARMA order selection, and
forecasting/intervals arrive in later phases.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import _musecore
from . import translate
from .boxcox import inv_box_cox

_TREND_OPTIONS = "rw/llt/srw/td"
_SEASONAL_OPTIONS = "none/linear/equal"


class PTS:
    def __init__(
        self,
        model: str = "ZZZ",
        lags: Optional[int] = None,
        orders=None,
        ic: str = "BICc",
        h: int = 0,
        holdout: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.lags = lags
        self._orders_arg = orders or {"ar": 0, "ma": 0, "select": False}
        self.ic = ic
        self.h = int(h)
        self.holdout = bool(holdout)
        self.verbose = bool(verbose)
        self._fit = None  # populated by fit()

    # ---- fit ------------------------------------------------------------
    def fit(self, y, X=None):
        y = np.asarray(y, dtype=float).ravel()
        if self.lags is None:
            raise ValueError(
                "lags must be supplied (pandas-index inference lands in a "
                "later phase)."
            )
        lags = int(self.lags)

        ar = int(np.atleast_1d(self._orders_arg.get("ar", 0))[0])
        ma = int(np.atleast_1d(self._orders_arg.get("ma", 0))[0])
        user_select = bool(self._orders_arg.get("select", False))

        criterion = translate.ic_to_engine(self.ic)

        # holdout split (before the lambda screen, matching R's order)
        y_full = y
        held = None
        if self.holdout and self.h > 0:
            held = y[len(y) - self.h :]
            y = y[: len(y) - self.h]
        self._u = self._prepare_u(X)

        # Box-Cox lambda screen for auto-lambda (power == "Z").  Mirrors
        # pts(): decomposition + Guerrero CV over [0, 2], then rewrite the
        # spec's power slot with the chosen numeric lambda (rounded to R's
        # 7-significant-digit string round-trip) and flag the +1 DoF.
        nm = len(self.model)
        power = self.model[: nm - 2].lower()
        lambda_screened = False
        model_str = self.model
        if power == "z" and len(y) >= 4:
            from .lambda_screen import guerrero_decomp_lambda
            best = guerrero_decomp_lambda(y, lags, lower=0.0, upper=2.0)
            best = float(f"{best:.7g}")  # match R format(..., digits = 7)
            model_str = _fmt_lambda_str(best) + self.model[nm - 2 :]
            lambda_screened = True

        arma_lags = [1]
        if user_select:
            from . import selector
            sel = selector.select_pts_arma(
                y, model_str, lags, ar, ma, arma_lags, self.ic,
                fit_structural=lambda spec: self._fit_structural(
                    spec, y, lags, criterion
                ),
            )
            # model_spec already carries the screened lambda + the selected
            # trend/seasonal letters; use it verbatim (do not re-slice).
            model_str = sel["model_spec"]
            ar = int(np.atleast_1d(sel["ar"])[0])
            ma = int(np.atleast_1d(sel["ma"])[0])
            arma_lags = sel["lags"]

        model_uc, lam = translate.pts_to_uc(model_str, arma_orders=(ar, ma))
        out = self._engine(y, self._u, model_uc, lam, lags, criterion, ar, ma,
                            arma_ident=False)
        if out.get("model") == "error":
            raise RuntimeError("muse engine returned an error for this spec.")

        self._post_process(out, y, y_full, held, lam, lambda_screened)
        self._orders = {"ar": ar, "ma": ma, "lags": arma_lags, "select": user_select}
        return self

    # ---- engine helpers -------------------------------------------------
    @staticmethod
    def _prepare_u(X):
        if X is None:
            return np.zeros((1, 2), dtype=float)
        u = np.asarray(X, dtype=float)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.shape[0] > u.shape[1]:
            u = u.T
        return u

    def _engine(self, y, u, model_uc, lam, lags, criterion, ar, ma, arma_ident):
        periods = lags / np.arange(1, max(1, lags // 2) + 1)
        rhos = np.ones_like(periods)
        return _musecore.ucomp(
            "all", y, u, model_uc, int(self.h), float(lam), 0.0, False,
            criterion, periods, rhos, self.verbose, False,
            np.array([-9999.9]), bool(arma_ident), np.array([-9999.99]),
            float(lags), _TREND_OPTIONS, _SEASONAL_OPTIONS,
            f"arma({ar},{ma})", 1, 0, -math.inf,
        )

    def _fit_structural(self, spec, y, lags, criterion):
        """Lightweight no-ARMA fit used by the order selector's Pass 1."""
        model_uc, lam = translate.pts_to_uc(spec, arma_orders=(0, 0))
        out = self._engine(y, self._u, model_uc, lam, lags, criterion, 0, 0,
                            arma_ident=False)
        if out.get("model") == "error":
            return None
        v = np.asarray(out["v"], dtype=float)
        return {
            "logLik": float(np.asarray(out["criteria"], dtype=float)[0]),
            "residuals": v[: len(y)],
            "n_p": int(np.asarray(out["coef"], dtype=float).size),
            "lambda": float(out["lambda"]),
            "lambdaEstimated": bool(out.get("lambdaEstimated", False)),
        }

    # ---- post-processing (mirror .pts_fit + pts()) ----------------------
    def _post_process(self, out, y_train, y_full, held, lam_in, lambda_screened):
        self._out = out
        self._y_train = y_train
        self._y_full = y_full
        self._held = held

        self._lambda = float(out["lambda"])
        self._model_uc = out["model"]
        # coef is the natural-scale parameter vector (variances multiplied
        # back by the concentrated MLE); out["p"] is the optimiser-space
        # vector.  R's coef()/$B uses out$coef -- so do we.
        self._p = np.asarray(out["coef"], dtype=float)
        self._par_names = list(out.get("parNames", []))[: self._p.size]
        self._model_label = translate.uc_to_pts(self._model_uc, self._lambda)

        crit = np.asarray(out["criteria"], dtype=float)
        self._logLik = float(crit[0]) if crit.size >= 1 else float("nan")

        # component matrix: engine returns (m x T); R reshapes to (T x m)
        comp_mt = np.asarray(out["comp"], dtype=float)  # (m, T)
        comp = comp_mt.T  # (T, m)
        names = out["compNames"].split("/")
        v = np.asarray(out["v"], dtype=float)
        self._comp, self._comp_names = _build_comp(comp, names, v)

        ns = len(y_train)
        error = self._comp[:, 0]
        fit_bc = self._comp[:, 1]
        self._residuals = error[:ns]
        self._fitted = inv_box_cox(fit_bc, self._lambda)[:ns]

        # MLE scale on the BC scale (matches .pts_fit / adam scaler())
        res = self._residuals[np.isfinite(self._residuals)]
        self._scale = (
            math.sqrt(float(np.sum(res ** 2)) / ns) if ns > 0 else float("nan")
        )

        # nParam = len(p) + (lambda estimated|screened) + (G/td drift slope)
        lam_dof = 1 if (bool(out.get("lambdaEstimated", False))
                        or lambda_screened) else 0
        td_dof = 1 if self._model_uc.startswith("td/") else 0
        self._nparam = int(self._p.size) + lam_dof + td_dof

        self._vcov = np.asarray(out["covp"], dtype=float) if "covp" in out else None
        self._fit = True

    # ---- properties -----------------------------------------------------
    @property
    def coef(self):
        return dict(zip(self._par_names, self._p)) if self._par_names else self._p

    @property
    def coef_values(self):
        return self._p

    @property
    def coef_names(self):
        return self._par_names

    @property
    def vcov(self):
        return self._vcov

    @property
    def fitted(self):
        return self._fitted

    @property
    def residuals(self):
        return self._residuals

    @property
    def actuals(self):
        return self._y_full

    @property
    def lambda_(self):
        return self._lambda

    @property
    def model_label(self):
        return self._model_label

    @property
    def model_uc(self):
        return self._model_uc

    @property
    def nobs(self):
        return int(np.sum(np.isfinite(self._y_train)))

    @property
    def n_param(self):
        return self._nparam

    @property
    def log_lik(self):
        return self._logLik

    @property
    def scale(self):
        return self._scale

    @property
    def sigma(self):
        n = self.nobs
        df = n - self._nparam
        if df <= 0:
            df = n
        res = self._residuals[np.isfinite(self._residuals)]
        return math.sqrt(float(np.sum(res ** 2)) / df)

    @property
    def comp(self):
        return self._comp, self._comp_names

    @property
    def orders(self):
        return self._orders

    # ---- information criteria (match R/greybox formulas) ----------------
    @property
    def aic(self):
        return -2.0 * self._logLik + 2.0 * self._nparam

    @property
    def bic(self):
        return -2.0 * self._logLik + math.log(self.nobs) * self._nparam

    @property
    def aicc(self):
        n, k = self.nobs, self._nparam
        denom = n - k - 1
        return self.aic + (2.0 * k * (k + 1)) / denom if denom > 0 else math.inf

    @property
    def bicc(self):
        n, k = self.nobs, self._nparam
        denom = n - k - 1
        return (
            self.bic + (math.log(n) * k * (k + 1)) / denom
            if denom > 0
            else math.inf
        )

    def __repr__(self):
        if not self._fit:
            return f"PTS(model={self.model!r}, lags={self.lags}) [unfitted]"
        return (
            f"PTS({self._model_label}, lambda={self._lambda:.3g}, "
            f"nParam={self._nparam}, logLik={self._logLik:.4g})"
        )


def _fmt_lambda_str(x: float) -> str:
    """Decimal (non-scientific) lambda string, trailing zeros stripped --
    matches R's format(x, scientific = FALSE, drop0trailing = TRUE) used when
    the screened lambda is written back into the model spec."""
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _build_comp(raw, names, v):
    """Port of .pts_build_comp: prepend Error (=v) and Fit (=sum of the
    selected structural columns); return (matrix, column_names).

    `raw` is (T x m) with column labels `names`.  Index logic follows the
    R original (translated from 1-based to 0-based)."""
    # ind (1-based in R): c(1, which Slope, which Seasonal)
    ind1 = [1]
    if "Slope" in names:
        ind1.append(names.index("Slope") + 1)
    if "Seasonal" in names:
        ind1.append(names.index("Seasonal") + 1)
    pos = max(ind1) + (1 if "Irregular" in names else 0)
    m = len(names)
    if pos > m:
        ind1 += list(range(pos + 1, m + 1))
    ind0 = [i - 1 for i in ind1]

    T = raw.shape[0]
    n_cols = 2 + len(ind0)
    comp = np.empty((T, n_cols), dtype=float)
    vv = np.asarray(v, dtype=float)
    if vv.size < T:  # pad (engine v is in-sample length; comp may include h)
        vv = np.concatenate([vv, np.full(T - vv.size, np.nan)])
    comp[:, 0] = vv[:T]               # Error
    comp[:, 2:] = raw[:, ind0]        # structural columns
    comp[:, 1] = comp[:, 2:].sum(axis=1)  # Fit = rowSum(structural)
    col_names = ["Error", "Fit"] + [names[i] for i in ind0]
    return comp, col_names
