# Changelog

## 0.1.0

* **Reliable parameter covariance for ill-conditioned models.** The
  observed-information Hessian behind the standard errors is now computed
  with central second differences and a per-parameter, magnitude-scaled
  step (was one-sided forward differences with a fixed step), and when that
  Hessian is indefinite or numerically singular at the optimum (a boundary /
  weakly-identified variance, or an ARMA φ≈θ near-cancellation) the
  covariance falls back to the OPG / BHHH estimator `Σ_t s_t s_tᵀ`, which is
  positive semidefinite by construction. Previously such models could return
  a non-PSD covariance and standard errors that swung by tens of percent
  under negligible numerical reordering. Point estimates, log-likelihood and
  information criteria are unchanged. Shares the R engine, so results match R
  `vcov(pts(...))`. (The OPG fallback is not yet available for models with
  external regressors, which keep the improved Hessian.)

* **Decoupled fit and forecast.** A fitted `PTS` object now caches the
  terminal state (final state, its covariance, the innovation variance, and
  the augmented-KF state for regressor models), so `predict()` / `forecast()`
  reuse it instead of re-filtering the whole series on every call — O(h)
  instead of O(n·m²). Forecasting a high-lag model is effectively instant
  after the fit (≈9× faster even at modest lags, more at high lags). Mirrors
  the R `pts()` behaviour, so `predict()` stays bit-for-bit consistent with R
  `forecast(pts(...))`.

* **Explanatory variables in forecasting (adam-style).** `fit()` now splits
  the regressor matrix into the in-sample and held-out parts; a fit with
  `h > 0, holdout=True` auto-forecasts using the held-out regressor rows.
  `predict(h, X=...)` accepts future regressor values for horizons beyond the
  holdout — the analogue of R's `forecast(model, h, newdata=...)`, and
  numerically identical to it.

* Initial public release: the Power / Trend / Seasonal (PTS) multiple-source-
  of-error state-space model, ported from the R `muse` package and sharing the
  same C++ engine. sklearn-style `PTS(...).fit(y).predict(h)` API with
  automatic Box-Cox power, trend, and seasonal selection.
