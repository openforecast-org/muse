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

import numpy as np

from .. import _musecore  # type: ignore[attr-defined]
from . import translate
from .boxcox import inv_box_cox

_TREND_OPTIONS = "rw/llt/srw/td"
_SEASONAL_OPTIONS = "none/linear/equal"

_NPARAM_ROWS = ["Estimated", "Provided"]
_NPARAM_COLS = ["nParamInternal", "nParamXreg", "nParamOccurrence",
                "nParamScale", "nParamAll"]


def _nparam_table(n_p, n_initial, lam_dof, n_xreg):
    """adam-style 2 x 5 parameter-count table (mirror of R .pts_nparam_table).

    Rows ["Estimated", "Provided"]; columns ["nParamInternal", "nParamXreg",
    "nParamOccurrence", "nParamScale", "nParamAll"].  One optimised parameter
    is the concentrated scale (-> nParamScale); the remaining structural
    parameters, the diffuse initials, and a free lambda fold into
    nParamInternal, exactly as smooth::adam does.  PTS has no occurrence model.
    """
    import pandas as pd

    m = np.zeros((2, 5), dtype=int)
    scale = 1 if n_p >= 1 else 0
    m[0, 0] = max(0, n_p - scale) + n_initial + lam_dof   # nParamInternal
    m[0, 1] = n_xreg                                      # nParamXreg
    m[0, 3] = scale                                       # nParamScale
    m[0, 4] = int(m[0, :4].sum())                         # nParamAll (Estimated)
    m[1, 4] = int(m[1, :4].sum())                         # nParamAll (Provided)
    return pd.DataFrame(m, index=_NPARAM_ROWS, columns=_NPARAM_COLS)


class PTS:
    def __init__(
        self,
        model: str = "ZZZ",
        lags: int | None = None,
        orders=None,
        ic: str = "AICc",
        outliers: str = "ignore",
        level: float = 0.99,
        lambda_estim: str = "likelihood",
        biasadj: bool = False,
        h: int = 0,
        holdout: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.lags = lags
        self._orders_arg = orders or {"ar": 0, "ma": 0, "select": False}
        self.ic = ic
        if outliers == "select":
            raise NotImplementedError("outliers='select' is not supported.")
        if outliers not in ("ignore", "use"):
            raise ValueError("outliers must be 'ignore' or 'use'.")
        self.outliers = outliers
        if lambda_estim not in ("likelihood", "guerrero", "decomp-guerrero"):
            raise ValueError(
                "lambda_estim must be 'likelihood', 'guerrero', "
                "or 'decomp-guerrero'."
            )
        self.lambda_estim = lambda_estim
        self.biasadj = bool(biasadj)
        if not (0 < level < 1):
            raise ValueError("level must be in (0, 1).")
        self.level = float(level)
        self.h = int(h)
        self.holdout = bool(holdout)
        self.verbose = bool(verbose)
        self._fit = None  # populated by fit()

    # ---- fit ------------------------------------------------------------
    def fit(self, y, X=None):
        from . import io
        y, self._index, inferred = io.parse_input(y)
        lags_in = self.lags if self.lags is not None else inferred
        if lags_in is None:
            raise ValueError(
                "lags could not be inferred; pass lags=<seasonal period>, or "
                "fit a pandas Series with a frequency-bearing DatetimeIndex."
            )
        lags = int(lags_in)
        self._lags = lags
        self._lags_all = lags / np.arange(1, max(1, lags // 2) + 1)

        # Normalise ARMA orders into per-lag vectors; the seasonal ARMA lag
        # comes from the top-level `lags` (not from `orders`).  ar/ma may be
        # scalars (non-seasonal) or length-2 (SARMA), matching R.
        ords = translate.orders_to_uc(self._orders_arg, lags)
        ar_v, ma_v, arma_lags = ords["ar"], ords["ma"], ords["lags"]
        user_select = ords["select"]

        criterion = translate.ic_to_engine(self.ic)

        # holdout split (before the lambda screen, matching R's order)
        y_full = y
        held = None
        if self.holdout and self.h > 0:
            held = y[len(y) - self.h :]
            y = y[: len(y) - self.h]
        self._index_full = self._index
        self._index_train = (self._index[: len(y)]
                             if self._index is not None else None)
        self._u = self._prepare_u(X)
        # Split off the held-out future regressors (adam-style: the auto-forecast
        # takes those rows of the data).  _u is k x N -> train k x n + fut k x h.
        self._has_x = X is not None
        self._u_future = None
        if self._has_x and self.holdout and self.h > 0:
            n_tr = len(y)            # y is already the training portion
            self._u_future = self._u[:, n_tr:]
            self._u = self._u[:, :n_tr]

        # Lower bound on lambda for the engine's joint-lambda BFGS (and the
        # screens): 0 for negatives, the zero floor for zero-containing series
        # (below it g(0) = -1/lambda is a pathological outlier / dropped at
        # lambda <= 0, making the likelihood incomparable), else unbounded.
        from .lambda_screen import lambda_zero_floor
        if np.any(y < 0):
            lambda_lower = 0.0
        elif np.any(y == 0):
            lambda_lower = lambda_zero_floor(y, 0.0)
        else:
            lambda_lower = -math.inf
        self._lambda_lower = lambda_lower

        # Box-Cox lambda selection for auto-lambda (power == "Z").  Mirrors
        # pts(): lambda_estim picks the method --
        #   "likelihood"      -> leave "Z" so the engine estimates lambda by ML
        #   "guerrero"        -> classical Guerrero CV on raw season blocks
        #   "decomp-guerrero" -> Guerrero on an msdecompose-smoothed trend
        # For the screens the power slot is rewritten with the chosen numeric
        # lambda (rounded to R's 7-sig-digit string round-trip) and the +1 DoF
        # flagged; for "likelihood" the engine flags lambdaEstimated itself.
        nm = len(self.model)
        power = self.model[: nm - 2].lower()
        lambda_screened = False
        model_str = self.model
        if power == "z" and self.lambda_estim != "likelihood" and len(y) >= 4:
            lower_screen = max(0.0, lambda_lower if math.isfinite(lambda_lower)
                               else 0.0)
            if self.lambda_estim == "guerrero":
                from .lambda_screen import guerrero_classic_lambda
                best = guerrero_classic_lambda(y, lags, lower=lower_screen,
                                               upper=2.0)
            else:
                from .lambda_screen import guerrero_decomp_lambda
                best = guerrero_decomp_lambda(y, lags, lower=lower_screen,
                                              upper=2.0)
            best = float(f"{best:.7g}")  # match R format(..., digits = 7)
            model_str = _fmt_lambda_str(best) + self.model[nm - 2 :]
            lambda_screened = True

        if user_select:
            from . import selector
            sel = selector.select_pts_arma(
                y, model_str, lags, ar_v, ma_v, arma_lags, self.ic,
                fit_structural=lambda spec: self._fit_structural(
                    spec, y, lags, criterion
                ),
            )
            # model_spec already carries the screened lambda + the selected
            # trend/seasonal letters; use it verbatim (do not re-slice).
            model_str = sel["model_spec"]
            ar_v = [int(v) for v in np.atleast_1d(sel["ar"])]
            ma_v = [int(v) for v in np.atleast_1d(sel["ma"])]
            arma_lags = [int(v) for v in np.atleast_1d(sel["lags"])]

        # outlier detection threshold (adam-style level -> two-sided z).
        # The lambda screen above already pins lambda to a number before the
        # engine runs, so R's joint-lambda + outlier workaround is moot here.
        from scipy.stats import norm
        outlier_z = 0.0 if self.outliers == "ignore" else float(
            norm.ppf((1 + self.level) / 2)
        )

        arma_tuple, irregular = translate.arma_spec(ar_v, ma_v, arma_lags)
        model_uc, lam = translate.pts_to_uc(model_str, arma_orders=arma_tuple)
        out = self._engine(y, self._u, model_uc, lam, lags, criterion,
                           irregular, arma_ident=False, outlier=outlier_z,
                           lambda_lower=lambda_lower)
        if out.get("model") == "error":
            raise RuntimeError("muse engine returned an error for this spec.")

        self._post_process(out, y, y_full, held, lam, lambda_screened)
        # store $orders like R: scalar ar/ma for non-seasonal, vectors for SARMA
        self._orders = {
            "ar": ar_v[0] if len(arma_lags) == 1 else ar_v,
            "ma": ma_v[0] if len(arma_lags) == 1 else ma_v,
            "lags": arma_lags, "select": user_select,
        }

        # Decoupled forecast cache: one forecastOnly pass populates the
        # absolute-scale terminal state (final state, covariance, innovation
        # variance, and the augmented-KF state for xreg) so predict() reuses it
        # without re-filtering the whole series.  Mirrors the R orchestration.
        self._aEnd = self._PEnd = self._betaAug = None
        self._innVar = -1.0
        try:
            fc = self._forecast_engine(int(self.h))
        except Exception:
            fc = {}
        # xreg with no future regressors: the h-step forecast can't be formed,
        # but the terminal-state cache is horizon-independent -- fetch it at h=0.
        if "aEnd" not in fc:
            try:
                fc = self._forecast_engine(0)
            except Exception:
                fc = {}
        if "aEnd" in fc:
            self._aEnd = np.asarray(fc["aEnd"], dtype=float)
            self._PEnd = np.asarray(fc["PEnd"], dtype=float)
            self._innVar = float(fc["innVar"])
            self._betaAug = np.asarray(fc["betaAug"], dtype=float)
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

    def _engine(self, y, u, model_uc, lam, lags, criterion, irregular,
                arma_ident, outlier=0.0, lambda_lower=-math.inf):
        periods = lags / np.arange(1, max(1, lags // 2) + 1)
        rhos = np.ones_like(periods)
        return _musecore.ucomp(
            "all", y, u, model_uc, int(self.h), float(lam), float(outlier),
            False, criterion, periods, rhos, self.verbose, False,
            np.array([-9999.9]), bool(arma_ident), np.array([-9999.99]),
            float(lags), _TREND_OPTIONS, _SEASONAL_OPTIONS,
            irregular, 1, 0, float(lambda_lower),
        )

    def _arma_candidate(self):
        o = self._orders
        lg = np.atleast_1d(o["lags"])
        ar = np.atleast_1d(o["ar"])
        ma = np.atleast_1d(o["ma"])
        if lg.size == 1:
            return f"arma({int(ar[0])},{int(ma[0])})"
        return f"arma({int(ar[0])},{int(ma[0])},{int(ar[1])},{int(ma[1])},{int(lg[1])})"

    def _forecast_engine(self, h, X=None):
        """forecastOnly: feed the fitted natural-scale coef back in and
        propagate h steps.  Mirrors .pts_forecast_inputs + forecastOnly.

        When the terminal-state cache (populated at fit time) is available,
        it is passed back so the engine reuses it instead of re-filtering the
        whole series -- O(h) instead of O(n*m^3).  For xreg models the future
        regressors come from `X` (predict(X=...)) or, failing that, the
        held-out rows captured at fit time (the auto-forecast)."""
        periods = np.asarray(self._lags_all, dtype=float)
        rhos = np.ones_like(periods)
        # xreg, adam-style: filter on the training regressors and append the
        # future rows so the engine forecasts with the right covariates.
        u = np.zeros((1, 2), dtype=float)
        if getattr(self, "_has_x", False):
            u = self._u
            if X is not None:
                fut = self._prepare_u(X)
            else:
                fut = getattr(self, "_u_future", None)
            if fut is not None and fut.shape[1] >= h:
                u = np.hstack([self._u, fut[:, :h]])
        aEnd = getattr(self, "_aEnd", None)
        cached = aEnd is not None and np.asarray(aEnd).size > 0
        return _musecore.ucomp(
            "forecastOnly", self._y_train, u, self._model_uc, int(h),
            float(self._lambda), 0.0, False, "aic", periods, rhos, False,
            False, np.asarray(self._p, dtype=float), False,
            np.array([-9999.99]), float(self._lags), _TREND_OPTIONS,
            _SEASONAL_OPTIONS, self._arma_candidate(), 1, 0, -math.inf,
            np.asarray(aEnd, dtype=float) if cached else np.zeros(0),
            np.asarray(getattr(self, "_PEnd", None), dtype=float)
                if cached else np.zeros((0, 0)),
            float(getattr(self, "_innVar", -1.0)) if cached else -1.0,
            np.asarray(getattr(self, "_betaAug", None), dtype=float)
                if cached else np.zeros(0),
        )

    def _forecast_paths(self, h, nsim, seed):
        """Forward simulation from the terminal state (original scale)."""
        periods = np.asarray(self._lags_all, dtype=float)
        rhos = np.ones_like(periods)
        u = np.zeros((1, 2), dtype=float)
        out = _musecore.ucomp(
            "simulate", self._y_train, u, self._model_uc, int(h),
            float(self._lambda), 0.0, False, "aic", periods, rhos, False,
            False, np.asarray(self._p, dtype=float), False,
            np.array([-9999.99]), float(self._lags), _TREND_OPTIONS,
            _SEASONAL_OPTIONS, self._arma_candidate(), int(nsim), int(seed),
            -math.inf,
        )
        return np.asarray(out["simPaths"], dtype=float)

    def predict(self, h, X=None, interval="prediction", level=0.95, side="both",
                cumulative=False, nsim=10000, seed=0, scenarios=False,
                biasadj=None):
        from .forecaster import forecast
        return forecast(self, h, X=X, interval=interval, level=level, side=side,
                        cumulative=cumulative, nsim=nsim, seed=seed,
                        scenarios=scenarios, biasadj=biasadj)

    def update(self, **overrides):
        """Re-fit on the same data with selected spec changes (sklearn-clone
        style).  e.g. m.update(h=24) or m.update(model="1LT")."""
        kw = dict(
            model=self.model, lags=self.lags, orders=self._orders_arg,
            ic=self.ic, lambda_estim=self.lambda_estim, biasadj=self.biasadj,
            h=self.h, holdout=self.holdout, verbose=self.verbose,
        )
        kw.update(overrides)
        return PTS(**kw).fit(self._y_full)

    def summary(self, level=0.95):
        """Coefficient table + variance proportions, replicating summary.pts.

        The concentrated-out variance (NaN on the inverse-Hessian diagonal)
        gets the analytical Gaussian-variance MLE SE |est|*sqrt(2/n); the
        joint vcov is patched (diag = se^2, cross terms zeroed) so the
        delta-method proportion SEs match R.  For a G/td trend the
        deterministic drift slope is injected as a Slope row (NaN SE).
        """
        from scipy.stats import norm
        est = self._p.astype(float)
        nm = list(self._par_names)
        n = self.nobs
        cv = (self._vcov.astype(float).copy()
              if (self._vcov is not None and np.ndim(self._vcov) == 2)
              else None)

        ses = np.full(est.size, np.nan)
        if cv is not None:
            k = cv.shape[0]
            cvdiag = np.diag(cv).copy()
            ses[:k] = np.sqrt(cvdiag)
            conc = np.where(np.isnan(cvdiag))[0]
            if conc.size and n > 0:
                se_conc = np.abs(est[conc]) * math.sqrt(2.0 / n)
                ses[conc] = se_conc
                for j, i in enumerate(conc):
                    cv[i, :] = 0.0
                    cv[:, i] = 0.0
                    cv[i, i] = se_conc[j] ** 2

        a = (1 - level) / 2
        z = norm.ppf([a, 1 - a])
        coef = {
            "names": list(nm), "estimate": est.copy(), "std_error": ses.copy(),
            "lower": est + ses * z[0], "upper": est + ses * z[1],
        }
        # deterministic drift slope row (G / td), inserted after Level
        det_slope = None
        if self._model_uc.startswith("td/") and "Slope" in self._comp_names:
            det_slope = float(self._comp[0, self._comp_names.index("Slope")])
            if "Level" in coef["names"]:
                pos = coef["names"].index("Level") + 1
                coef["names"].insert(pos, "Slope")
                for key, val in (("estimate", det_slope), ("std_error", np.nan),
                                 ("lower", np.nan), ("upper", np.nan)):
                    coef[key] = np.insert(coef[key], pos, val)

        # variance proportions (exclude AR/MA, Beta, Damping, outliers)
        is_var = np.array([
            not (n.startswith(("AR(", "SAR(", "MA(", "SMA(", "Beta"))
                 or n == "Damping" or _is_outlier(n))
            for n in nm
        ])
        var_vals = est[is_var]
        S = float(np.sum(var_vals))
        props = var_vals / S if (var_vals.size and S > 0) else var_vals
        prop_ses = np.full(var_vals.size, np.nan)
        if var_vals.size > 1 and cv is not None and S > 0:
            var_idx = np.where(is_var)[0]
            if var_idx.max() < cv.shape[0]:
                Sv = cv[np.ix_(var_idx, var_idx)]
                if np.all(np.isfinite(Sv)):
                    L = var_vals.size
                    J = (np.eye(L) - np.ones((L, 1)) @ props[None, :]) / S
                    prop_var = np.diag(J @ Sv @ J.T)
                    prop_ses = np.sqrt(np.maximum(0.0, prop_var))

        return {
            "model": self._model_label, "lambda": self._lambda,
            "nobs": n, "n_param": self._nparam, "sigma": self.sigma,
            "logLik": self._logLik,
            "ic": {"AIC": self.aic, "BIC": self.bic, "AICc": self.aicc,
                   "BICc": self.bicc},
            "coefficients": coef,
            "proportions": {"names": [x for x, kk in zip(nm, is_var, strict=False) if kk],
                            "proportion": props, "std_error": prop_ses},
        }

    # ---- plotting (reuses smooth's plot_adam via a duck-typed adapter) --
    def plot(self, which=(1, 2, 4, 6), level=0.95, legend=False, lowess=True,
             **kwargs):
        """Diagnostic plots, reusing smooth.adam's plot_adam on an adapter
        that exposes the attributes it duck-types on.  Plots 1-7,9 (incl. the
        default 1,2,4,6) need only fitted/residuals/scale/distribution; the
        states plot (12) is best-effort."""
        from smooth.adam_general.core.plotting import plot_adam
        return plot_adam(_PlotAdapter(self), which=list(np.atleast_1d(which)),
                         level=level, legend=legend, lowess=lowess, **kwargs)

    # ---- diagnostics ----------------------------------------------------
    def rstandard(self):
        s = self.sigma
        e = self._residuals
        if s == 0 or not math.isfinite(s):
            return np.full_like(e, np.nan)
        return e / s

    def rstudent(self):
        # Same conservative approximation R uses (no per-obs leverage in a
        # state-space model).
        return self.rstandard()

    def point_lik(self, log=True):
        from scipy.stats import norm
        s = self.sigma
        e = self._residuals
        if s == 0 or not math.isfinite(s):
            return np.full_like(e, np.nan)
        return norm.logpdf(e, 0.0, s) if log else norm.pdf(e, 0.0, s)

    def confint(self, level=0.95):
        from scipy.stats import norm
        est = self._p
        cv = self._vcov
        if cv is not None and np.ndim(cv) == 2 and cv.shape[0] == est.size:
            ses = np.sqrt(np.diag(cv))
        else:
            ses = np.full(est.size, np.nan)
        a = (1 - level) / 2
        z = norm.ppf([a, 1 - a])
        lower = est + ses * z[0]
        upper = est + ses * z[1]
        return {
            "names": self._par_names,
            "lower": lower,
            "upper": upper,
            "level_labels": [f"{100*a:.1f} %", f"{100*(1-a):.1f} %"],
        }

    def accuracy(self, holdout=None):
        import greybox
        if holdout is None:
            holdout = self._held
        if holdout is None:
            raise ValueError(
                "No holdout provided and the model carries none; fit with "
                "holdout=True or pass holdout explicitly."
            )
        holdout = np.asarray(holdout, dtype=float)
        pred = np.asarray(self.predict(len(holdout), interval="none").mean,
                          dtype=float)
        # R's accuracy.pts scales against actuals(object) == the in-sample
        # training series (object$data), not the full series.
        return greybox.measures(holdout=holdout, forecast=pred,
                                actual=np.asarray(self._y_train, dtype=float))

    def simulate(self, nsim=1, seed=0):
        """In-sample replay from the initial state (command='simulateInit').
        Returns an (nobs x nsim) original-scale matrix."""
        periods = np.asarray(self._lags_all, dtype=float)
        rhos = np.ones_like(periods)
        u = np.zeros((1, 2), dtype=float)
        out = _musecore.ucomp(
            "simulateInit", self._y_train, u, self._model_uc, int(self.nobs),
            float(self._lambda), 0.0, False, "aic", periods, rhos, False,
            False, np.asarray(self._p, dtype=float), False,
            np.array([-9999.99]), float(self._lags), _TREND_OPTIONS,
            _SEASONAL_OPTIONS, self._arma_candidate(), int(nsim), int(seed),
            -math.inf,
        )
        return np.asarray(out["simPaths"], dtype=float)

    def _fit_structural(self, spec, y, lags, criterion):
        """Lightweight no-ARMA fit used by the order selector's Pass 1."""
        model_uc, lam = translate.pts_to_uc(spec, arma_orders=(0, 0))
        out = self._engine(y, self._u, model_uc, lam, lags, criterion,
                            "arma(0,0)", arma_ident=False,
                            lambda_lower=getattr(self, "_lambda_lower",
                                                 -math.inf))
        if out.get("model") == "error":
            return None
        v = np.asarray(out["v"], dtype=float)
        return {
            "logLik": float(np.asarray(out["criteria"], dtype=float)[0]),
            "residuals": v[: len(y)],
            "n_p": int(np.asarray(out["coef"], dtype=float).size),
            "lambda": float(out["lambda"]),
            "lambdaEstimated": bool(out.get("lambdaEstimated", False)),
            "n_initial": int(out.get("nInitial", 0)),
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

        # nParam: an adam-style breakdown table (mirror of smooth::adam and of
        # the R pts() $nParam).  A 2 x 5 DataFrame, rows ["Estimated",
        # "Provided"], columns ["nParamInternal", "nParamXreg",
        # "nParamOccurrence", "nParamScale", "nParamAll"].  n_param returns the
        # [Estimated, nParamAll] total, so the IC degrees of freedom is
        # unchanged from a plain scalar -- only the presentation is richer.
        #
        # Estimated initials n_initial = ns(0)+ns(1)+ns(2) (level/slope + cycle
        # + seasonal) come from the engine (lags-driven, multi-seasonal-correct,
        # stationary ARMA states excluded) and are folded into nParamInternal,
        # exactly as adam does.  They already include the G/td drift (= initial
        # slope), so no separate td term.  The single concentrated variance is
        # nParamScale; regressors are nParamXreg; PTS has no occurrence model.
        # The engine adds the same quantities to its own selection k (kFor in
        # BSMclass::estim), so selection and reporting agree.
        lam_dof = 1 if (bool(out.get("lambdaEstimated", False))
                        or lambda_screened) else 0
        n_initial = int(out.get("nInitial", 0))
        # Real regressor count: the engine treats the (1, 2) all-zero dummy as
        # "no xreg" (resizes it away), so mirror that rule here.
        u = self._u
        n_xreg = 0 if (u is None or tuple(u.shape) == (1, 2)) else int(u.shape[0])
        self._nparam_table = _nparam_table(
            int(self._p.size), n_initial, lam_dof, n_xreg)
        self._nparam = int(self._nparam_table.loc["Estimated", "nParamAll"])

        self._vcov = np.asarray(out["covp"], dtype=float) if "covp" in out else None

        # outliers detected by the engine: (nDetected x 2) = (type, time0).
        det = out.get("typeOutliers")
        rows = []
        if det is not None:
            det = np.atleast_2d(np.asarray(det, dtype=float))
            if det.size and det.shape[1] == 2:
                kinds = ["AO", "LS", "SC"]
                for typ, t0 in det:
                    rows.append({"time": int(t0) + 1, "type": kinds[int(typ)]})
        self._outliers_detected = rows
        self._fit = True

    # ---- properties -----------------------------------------------------
    @property
    def coef(self):
        return dict(zip(self._par_names, self._p, strict=False)) if self._par_names else self._p

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
        from . import io
        return io.wrap(self._fitted, self._index_train)

    @property
    def residuals(self):
        from . import io
        return io.wrap(self._residuals, self._index_train)

    @property
    def actuals(self):
        from . import io
        return io.wrap(self._y_full, self._index_full)

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
        """Total estimated degrees of freedom ([Estimated, nParamAll])."""
        return self._nparam

    @property
    def n_param_table(self):
        """adam-style 2 x 5 breakdown of the parameter count (DataFrame)."""
        return self._nparam_table

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

    @property
    def outliers_detected(self):
        return self._outliers_detected

    def outlierdummy(self, level=0.999, type="rstandard"):
        from scipy.stats import norm
        r = self.rstandard() if type == "rstandard" else self.rstudent()
        q = norm.ppf(level)
        ids = np.where(np.abs(np.asarray(r, dtype=float)) > q)[0]
        return {"id": ids, "statistic": (-q, q), "type": type, "level": level}

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


class _OutlierDummyResult:
    """Mirror of smooth's outlierdummy return (it reads `.statistic`)."""
    def __init__(self, statistic, ids, type_, level):
        self.statistic = np.asarray(statistic, dtype=float)
        self.id = ids
        self.type = type_
        self.level = level


class _PlotAdapter:
    """Duck-typed view of a fitted PTS exposing the attribute names that
    smooth.adam_general.core.plotting.plot_adam reads."""
    def __init__(self, m):
        self._m = m
        self.fitted = np.asarray(m._fitted, dtype=float)
        # in-sample actuals (training); the holdout lives in holdout_data,
        # matching smooth's plot convention (so fitted/actuals align).
        self.actuals = np.asarray(m._y_train, dtype=float)
        self.residuals = np.asarray(m._residuals, dtype=float)
        self.scale = m._scale
        self.distribution_ = "dnorm"
        self.model_name = m._model_label
        self.is_combined = False
        self.holdout_data = (np.asarray(m._held, dtype=float)
                             if m._held is not None else None)
        self._config = {}
        # states (n_states x T+1): the structural state columns of comp
        # (Level / Slope / Seasonal), excluding Error / Fit / Irregular.
        comp, names = m._comp, m._comp_names
        struct = [i for i, n in enumerate(names)
                  if n not in ("Error", "Fit", "Irregular")
                  and not _is_outlier(n)
                  and not n.startswith(("AR(", "MA(", "SAR(", "SMA(", "Beta"))]
        S = comp[:, struct].T if struct else np.zeros((1, comp.shape[0]))
        self.states = np.hstack([S[:, :1], S])
        self._components = {
            "model_is_trendy": "Slope" in names,
            "components_number_ets_seasonal": 1 if "Seasonal" in names else 0,
            "components_number_arima": 0,
        }

    def rstandard(self):
        return np.asarray(self._m.rstandard(), dtype=float)

    def rstudent(self):
        return np.asarray(self._m.rstudent(), dtype=float)

    def outlierdummy(self, level=0.999, type="rstandard"):
        d = self._m.outlierdummy(level=level, type=type)
        return _OutlierDummyResult(d["statistic"], d["id"], d["type"],
                                   d["level"])


def _is_outlier(name: str) -> bool:
    import re
    return bool(re.match(r"^(AO|LS|SC)[0-9]+$", name))


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
