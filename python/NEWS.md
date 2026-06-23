# Changelog

## 0.1.0

* **Outlier detection (`outliers="use"`) fixed.** The disturbance smoother
  no longer zeroes the auxiliary residual of any component whose variance is
  far below the largest (a single matrix `pinv()` tolerance was discarding it),
  so injected spikes are detected; and a detected, highly-significant outlier
  is no longer dropped merely because an over-fit baseline has a "better"
  information criterion (acceptance now follows the per-outlier significance
  test). Shared C++ engine, so identical to R.

* **Estimation gradient corrected; better convergence.** The analytic
  log-likelihood gradient used by the optimiser was wrong for structural
  models (it mixed ratio- and absolute-scale `Q`/`H` when building the
  finite-difference `dQ`, blowing up by ~1/innVariance for small
  concentrated variances, plus a normalisation error on seasonal models),
  and the optimiser could stall at a non-stationary point when its BFGS
  direction stopped being a descent direction. Both are fixed (shared C++
  engine, so identical to R): structural models now converge to materially
  higher likelihoods and the old "degenerate" fits (optimiser quitting
  after ~2 iterations with near-zero variances) are resolved. Matches R
  `pts()` exactly.

* **Reliable parameter covariance for ill-conditioned models.** The
  observed-information Hessian behind the standard errors is now computed
  with central second differences and a per-parameter, magnitude-scaled
  step (was one-sided forward differences with a fixed step), and when that
  Hessian is indefinite, numerically singular, or non-finite at the optimum
  (a boundary / weakly-identified variance, or an ARMA φ≈θ near-cancellation)
  the covariance falls back to the OPG / BHHH estimator `Σ_t s_t s_tᵀ`, which
  is positive semidefinite by construction. Previously such models could return
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
