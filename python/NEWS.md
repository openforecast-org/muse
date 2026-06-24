# Changelog

## 0.1.0

* **Information criteria now charge for the estimated structural initials, and
  a new `n_param_table` gives an adam-style breakdown.** The initial level,
  slope (local / global / damped trend), cycle, and seasonal states are diffuse
  / profiled out and so are genuine degrees of freedom; they were previously
  free in the IC, letting BICc / AICc under-penalise flexible seasonal and trend
  shapes. The estimated-initials count `ns(0)+ns(1)+ns(2)` is read from the
  engine so it is driven by `lags` -- correct for multi-seasonal
  `lags=[m1, m2, ...]` with no hard-coded period sizes; the stationary ARMA
  states are excluded (the ARMA coefficients are still counted). The engine adds
  the same quantity to its own selection criterion, so `PTS(model="ZZZ")`
  selection and the reported `n_param` agree. `n_param_table` is a 2 x 5
  DataFrame mirroring R's `smooth::adam` `$nParam` (rows `Estimated` /
  `Provided`; columns `nParamInternal`, `nParamXreg`, `nParamOccurrence`,
  `nParamScale`, `nParamAll`); `n_param` returns the `[Estimated, nParamAll]`
  total. The initials fold into `nParamInternal`; the concentrated variance is
  `nParamScale`. `coef` / `vcov` cover only the estimated coefficients, so
  `len(coef)` is smaller than `n_param`. Shared C++ engine, identical to R.

* **Point forecasts are now the conditional MEAN under a Box-Cox transform**,
  not the median. Back-transforming the BC-scale point forecast gives the
  median; for `lambda < 1` the convex inverse makes the median below the mean,
  so forecasts under-reported (~8% on a `lambda = 0.43` series). A second-order
  bias correction `mean ~= g^{-1}(mu) * (1 + 0.5*var*(1-lambda)/(1+lambda*mu)^2)`
  is now applied (matching `forecast::InvBoxCox(biasadj=True)`). Interval
  quantiles are unchanged. Identical to R.

* **Zero-containing series can now be variance-stabilised** (and forecast
  non-negatively). Previously any zero forced `lambda = 1` (raw scale) -- the
  Guerrero lambda screen bailed to 1 on `any(y <= 0)` and a non-1 lambda gave a
  NaN log-likelihood -- so intermittent / zero-heavy series were fit on the raw
  scale and routinely forecast negative. Now the BCnorm density allows `y == 0`
  for `lambda > 0` (rejecting only the undefined `y <= 0, lambda <= 0` corner),
  and the lambda screen runs on zero series (disqualifying only on negatives)
  with a floor `lambda >= log(2)/log(max(y))`. A power transform in (0, 1) fits
  such series far better and makes forecasts non-negative. Shared C++ engine;
  the R and Python lambda screens stay in parity.

* **Concentrated variance can no longer come out negative.** In the
  augmented Kalman likelihood the concentrated innovation variance is the
  post-projection residual sum of squares divided by `n` -- a sum of
  squares, so `>= 0`. It was formed as `v2F - sn'Sn^{-1}sn`, and when the
  augmented states (diffuse initial states + regressors) nearly interpolate
  the data those two terms are huge and nearly equal, so the subtraction
  loses all precision (catastrophic cancellation) and could flip negative,
  giving negative variances and NaN standard errors. The RSS is now formed
  directly as the sum of squared GLS residuals `Sum (v_t - V_t·beta)^2/F_t`,
  which is structurally non-negative, so the artifact cannot occur. Shared
  C++ engine, so identical to R.

* **Forecasts no longer blow up on a collapsed variance.** When a
  disturbance variance is driven to ~0, `exp(2*p)` underflowed to exactly 0
  in the state-space matrices, zeroing the Kalman innovation variance and
  producing `NaN` terminal states and explosive (~1e12) forecasts. Variance
  log-parameters are now floored at `var >= ~1e-10` in the matrix build.
  Shared C++ engine, so identical to R.

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
