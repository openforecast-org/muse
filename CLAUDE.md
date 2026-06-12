# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package overview

`muse` ("Multiple Unobserved Sources of Error") is an R package implementing the PTS (Power / Trend / Seasonal) state-space model for time-series analysis and forecasting. It depends on `Rcpp`, `RcppArmadillo`, `greybox`, and `smooth`. License: LGPL-2.1.

## ARCHITECTURE.md — primary reference

`ARCHITECTURE.md` in the package root is the authoritative source for how this
codebase is structured.  **Read it first** before exploring files or making changes.
It covers the full call graph for every task (estimation, selection, forecasting,
simulation, diagnostics), all C++ ↔ R data structures, coupling rules, and key
invariants (BC scale, variance parameterisation, concentrated likelihood, λ DoF).

**Keep it up to date.**  After any code change that affects call flows, data
structures, parameter naming, or cross-file coupling, update the relevant section of
`ARCHITECTURE.md` in the same commit.  If a coupling rule changes (e.g. a new field
must be added in two places), update the "Coupling rules" table.  If a new invariant
is discovered, add it to "Key invariants".  Stale documentation is worse than none.

## Common commands

R package development uses `R CMD` and `devtools`/`roxygen2`. From the package root:

- Regenerate `NAMESPACE` and `man/*.Rd` from roxygen comments: `Rscript -e 'roxygen2::roxygenise()'` (or `devtools::document()`). Required after editing any `#'` doc block or `@export`.
- Regenerate `R/RcppExports.R` and `src/RcppExports.cpp` after adding/removing `// [[Rcpp::export]]` functions in C++: `Rscript -e 'Rcpp::compileAttributes()'`.
- Build the package tarball: `R CMD build .`
- Check the package (run before committing significant changes): `R CMD check muse_*.tar.gz` (or `devtools::check()`).
- Install locally: `R CMD INSTALL .` (or `devtools::install()`).
- Run the full test suite: `Rscript -e 'devtools::test()'` (uses `testthat`; entry point is `tests/testthat.R`).
- Run a single test file: `Rscript -e 'devtools::test(filter="PTS")'` (matches `tests/testthat/test_PTS.R` and `test_PTSsetup.R`).
- Quick reload during development: `Rscript -e 'devtools::load_all()'` — recompiles C++ as needed.

The C++ side is built via `src/Makevars` (links LAPACK/BLAS/Fortran). Stale `*.o` / `muse.so` artifacts in `src/` can cause confusing build errors — delete them and rebuild if the C++ behavior seems mismatched with the source.

## Architecture

Full call-flow and data-structure documentation is in `ARCHITECTURE.md`.  The summary below covers what an agent needs to make safe changes.

### How it fits together

`pts()` (`R/pts.R`) is the sole user-facing function.  It calls `.pts_fit()` (`R/pts-internals.R`), which translates the 3-letter model string to a UC string (`R/pts-translate.R`), marshals arguments, and invokes `UCompC()` — the single C++ entry point (`src/musecpp2R.cpp`).  `UCompC()` dispatches on a `command` argument (`"all"` for full estimation, `"forecastOnly"`, `"simulate"`, …) and returns a named list that `.pts_fit()` post-processes into the `pts` S3 object.  S3 methods live in `R/methods.R`, `R/pts-summary.R`, `R/pts-methods-accessors.R`, `R/pts-methods-diagnostics.R`, `R/pts-accuracy.R`, `R/pts-simulate.R`, and `R/pts-confint.R`.  Dispatch for `plot`, `AIC`/`BIC`, and several greybox generics falls through the `c("pts", "smooth")` class chain — no local implementations needed.

### R file map

| File | What it contains |
|------|-----------------|
| `R/pts.R` | `pts()` entry point; assembles the returned object |
| `R/pts-internals.R` | `.pts_fit()`, `.pts_uc_inputs()`, `.pts_call_uc()`, `.pts_parse_data()`, `.pts_wrap_*()`, `.pts_build_comp()`, `.pts_forecast_inputs()`, `.inv_box_cox()` |
| `R/pts-translate.R` | `pts_to_uc()`, `uc_to_pts()`, `uc_to_arma()`, `.pts_ic_to_engine()` |
| `R/methods.R` | `print.pts`, `fitted`, `residuals`, `coef`, `vcov`, `nobs`, `logLik`, `predict`, `forecast.pts` |
| `R/pts-summary.R` | `summary.pts`, `print.summary.pts`, `as.data.frame.summary.pts` |
| `R/pts-methods-accessors.R` | `sigma`, `nparam`, `actuals`, `modelType`, `lags`, `orders`, `errorType` |
| `R/pts-methods-diagnostics.R` | `rstandard`, `rstudent`, `pointLik`, `outlierdummy` |
| `R/pts-accuracy.R` | `accuracy.pts`, `accuracy.pts.forecast` |
| `R/pts-simulate.R` | `simulate.pts` |
| `R/pts-confint.R` | `confint.pts` (Wald intervals) |
| `R/pts-update.R` | `update.pts` |
| `R/RcppExports.R` | Auto-generated — do not edit; regenerate with `Rcpp::compileAttributes()` |

### C++ file map

Header-only, all included from `src/musecpp2R.cpp`:

| File | What it contains |
|------|-----------------|
| `src/musecpp2R.cpp` | `UCompC()` Rcpp bridge; marshals SEXP ↔ `MuseInputs`/`MuseOutputs` |
| `src/musecore.h` | `MuseInputs`, `MuseOutputs` structs; `runMuseCommand()` dispatch |
| `src/PTSmodel.h` | `BSMclass` — model matrices, likelihood, ident, profile-lambda, components |
| `src/SSpace.h` | `SSmodel` — Kalman filter/smoother, quasi-Newton optimiser, forecast |
| `src/boxcox.h` | `BoxCox()`, `invBoxCox()`, `testBoxCox()` — thresholds must match `.inv_box_cox()` in R |
| `src/optim.h` | BFGS-style optimiser used by `SSmodel::estim()` |
| `src/ARMAmodel.h` | ARMA irregular component |
| `src/ARIMASSmodel.h` | Included but unused (dead code) |
| `src/stats.h`, `src/DJPTtools.h` | Statistical helpers and utilities |

### Coupling rules — when you change one thing, also change these

**Adding a new output field from C++ to R:**
1. Populate it in `MuseOutputs` and pack it in `musecpp2R.cpp`.
2. Unpack it in `.pts_fit()` (`R/pts-internals.R`) and add it to the returned list.
3. Store it on the `pts` object in `pts()` (`R/pts.R`).
4. If needed in `forecast.pts`, also unpack it in `.pts_forecast_inputs()`.

**Adding a new input field from R to C++:**
1. Add it to the list returned by `.pts_uc_inputs()` and `.pts_forecast_inputs()`.
2. Add it to the `UCompC()` signature in `src/musecpp2R.cpp` and unpack into `MuseInputs`.
3. Regenerate `R/RcppExports.R` and `src/RcppExports.cpp` with `Rcpp::compileAttributes()`.

**Changing Box-Cox branches:** the R-side `.inv_box_cox()` (`R/pts-internals.R`) and the C++ `BoxCox()`/`invBoxCox()` (`src/boxcox.h`) plus `bcnormBoxCox()`/`bcnormLogJac()` (`src/bcnorm.h`) must use identical branches.  Current convention is exact equality at the two singular points: `λ == 0` → log; `λ == 1` → identity; otherwise the general `(y^λ − 1)/λ` formula.  Do not reintroduce threshold-based shortcuts — they make AIC discontinuous in λ.

**Changing parameter names:** `BSMclass::parLabels()` in `PTSmodel.h` sets names; the S3 methods in `R/methods.R` and `R/pts-summary.R` pattern-match on them (`grepl("^AR\\("`, `"^Beta"`, `"^Damping"`, `"^Irregular"`).  Update both sides.

**Changing `nParam` logic:** the count is assembled in `pts()` as `length(res$p) + as.integer(lambdaEstimated)`.  The C++ engine sets `lambdaEstimated`; the R side must not add any further adjustment.

### Key invariants

- **BC scale vs original scale.** `$residuals`, `$comp`, and innovations are on the Box-Cox scale (additive).  `$fitted`, `$forecast$mean`, and holdout comparisons are on the original scale (back-transformed).  Prediction intervals are computed by endpoint-transforming the BC-scale ±z·se bounds — not by transforming a symmetric original-scale interval.
- **Concentrated variances.** Some variance parameters are concentrated out analytically; they appear in `$B` but have `NaN` rows/columns in `$vcov`.  `$nParam` correctly counts them for ICs.
- **Regressor matrix orientation.** User supplies (n × k); internally and in C++ it is (k × n).  The transpose happens once in `.pts_parse_data()` for training data and once in `.pts_forecast_inputs()` for `newdata`.
- **Harmonic periods.** `lagsAll = lags / (1 : floor(lags/2))`.  The C++ engine may select a subset, but `lagsAll` always stores the full candidate set that was passed in.

### Documentation generation

`man/` is generated from roxygen comments — never edit `.Rd` files by hand. Shared roxygen fragments live in `man-roxygen/` (e.g. `authors.R`, `keywords.R`) and are included via `@template authors`. `DESCRIPTION` sets `Roxygen: list(old_usage = TRUE)`, so generated usage sections use the legacy style — don't switch it.
