# Changelog

## 0.1.0

* **Decoupled fit and forecast.** A fitted `PTS` object now caches the
  terminal state (final state, its covariance, the innovation variance, and
  the augmented-KF state for regressor models), so `predict()` / `forecast()`
  reuse it instead of re-filtering the whole series on every call — O(h)
  instead of O(n·m²). Forecasting a high-lag model is effectively instant
  after the fit (≈9× faster even at modest lags, more at high lags), and works
  the same way with explanatory variables. Mirrors the R `pts()` behaviour, so
  `predict()` stays bit-for-bit consistent with R `forecast(pts(...))`.

* Initial public release: the Power / Trend / Seasonal (PTS) multiple-source-
  of-error state-space model, ported from the R `muse` package and sharing the
  same C++ engine. sklearn-style `PTS(...).fit(y).predict(h)` API with
  automatic Box-Cox power, trend, and seasonal selection.
