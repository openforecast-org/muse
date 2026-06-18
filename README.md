# muse - Multiple Unobserved Sources of Error

![muse logo](man/figures/muse-purple-light-web.png){width=150px}

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1)

<!-- badges: start -->

[![R-CMD-check](https://github.com/config-i1/muse/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/config-i1/muse/actions/workflows/R-CMD-check.yaml)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/muse)](https://cran.r-project.org/package=muse)
[![Downloads](http://cranlogs.r-pkg.org/badges/muse)](https://cran.r-project.org/package=muse)
<!-- badges: end -->

**muse** is an R package implementing the **PTS** (*Power / Trend / Seasonal*) state-space family of models for time-series analysis and for forecasting. The estimation engine is written in C++ (Rcpp & RcppArmadillo), wrapping a Kalman filter/smoother around a Multiple Source of Error (MSOE) model whose components are selected analogously to the
ETS taxonomy.

The package shares conventions with [`smooth`](https://github.com/config-i1/smooth) and [`greybox`](https://github.com/config-i1/greybox): the returned object inherits from `c("pts", "smooth")` so generics such as `forecast()`, `accuracy()`, `plot()`, `AIC()`, and `BIC()` work out of the box.

## Installation

The package is not yet on CRAN.  Install the development version directly from GitHub:

```r
# install.packages("remotes")
remotes::install_github("config-i1/muse")
```

Once released on CRAN:

```r
install.packages("muse")
```

## What's inside

The user-facing entry point is `pts()`.  In a single call it can:

- **Specify a PTS model** with a compact three-character string `"PTS"` —
  *Power* / *Trend* / *Seasonal* in that order (e.g. `"ZZZ"`, `"0LT"`, `"0.5GD"`)
  - `P` — *power* (Box-Cox λ): a numeric value or `"Z"` to estimate it jointly with the state-space parameters.
  - `T` — *trend*: `N` none, `L` local, `D` damped, `G` global, or `Z` for automatic selection.
  - `S` — *seasonal*: `N` none, `D` discrete, `T` trigonometric (harmonic seasonality), or `Z` for automatic selection.
- **Auto-select the best model** by AIC / BIC / AICc / BICc on the irregular component (`select = TRUE` or `Z` letters in the model string).
- **Estimate jointly with ARMA / SARMA** noise on the irregular component (`arma = c(p, q)` or a list).
- **Handle missing values** in the response series — they are filtered through the Kalman recursion natively, no imputation required.
- **Detect outliers** of type *AO* (additive), *LS* (level shift), and *SC* (slope change) during estimation via `outliers = "use"` with a user-supplied confidence `level` (e.g. `0.99`).  Detected outliers are reported in the fitted object and added as dummy regressors.
- **Take external regressors** through the `data` argument (a `data.frame` / `ts` / `matrix` whose first column is the response).
- **Produce forecasts** with prediction, confidence, or simulated intervals through `forecast()`.
- **Simulate trajectories** from the fitted model with `simulate()` (in-sample replay from the initial state, forward simulation for forecasts).
- **Hold out and score** automatically via `holdout = TRUE` plus `h = ...`.

The full method list includes `print`, `summary`, `plot`, `coef`, `vcov`, `confint`, `sigma`, `nobs`, `nparam`, `logLik`, `AIC`, `BIC`, `fitted`, `residuals`, `rstandard`, `rstudent`, `pointLik`, `accuracy`, `actuals`, `modelType`, `lags`, `orders`, `errorType`, `outlierdummy`, and `update`.


## Quick start

```r
library(muse)

# Fit a model with automatic Box-Cox, trend, seasonal,
# holding out the last 12 observations and forecasting them back.
model <- pts(AirPassengers, model = "ZZZ", h = 12,
             holdout = TRUE, ic = "AICc")
summary(model)
plot(forecast(model, h = 12))

# With engine-side outlier detection at the 99% confidence level
model_out <- pts(AirPassengers, model = "ZZZ", h = 12,
                 outliers = "use", level = 0.99)
model_out$outliersDetected
```

## Reporting issues

Bug reports and feature requests are welcome at the
[issue tracker](https://github.com/config-i1/muse/issues).
