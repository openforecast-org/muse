# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package overview

`muse` ("Multiple Unobserved Sources of Error") is an R package implementing the PTS (Power / Trend / Seasonal) state-space model for time-series analysis and forecasting. It depends on `Rcpp`, `RcppArmadillo`, `greybox`, and `smooth`. License: LGPL-2.1.

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

The package exposes a single user-facing model family — **PTS** — implemented as a thin R wrapper over an Armadillo C++ backend. An internal **MSOE** (general unobserved-components) layer sits between PTS and the C++ engine but is not exported.

### Public API

- **PTS** (`R/PTSfunctions.R`, `R/PTSS3functions.R`) — "Power/Trend/Seasonal" exponential-smoothing-style models. User specifies a three-letter model string (e.g. `"ZZZ"`, `"0NT"`, `"1LD"`) covering Power (Z or a numeric Box-Cox lambda), Trend (Z/N/L/G/D), and Seasonal (Z/N/D/T) components. Top-level entry points are `PTS()` (estimate + validate + components), `PTSforecast()` (estimate + forecast only, faster), and `PTSsetup()` (build the input object). Translators `PTS2modelUC`, `modelUC2PTS`, `modelUC2arma` map between the PTS spec and the underlying UC representation. S3 methods: `print.PTS`, `summary.PTS`, `plot.PTS`, `fitted.PTS`, `residuals.PTS`.

### Internal engine

- **MSOE** (`R/MSOEfunctions.R`) — `MSOEsetup`, `MSOE`, `MSOEestim`. **Not exported.** PTS calls these internally with the richer UC model-string grammar `"trend/cycle/seasonal/irregular"` (e.g. `"llt/none/equal/arma(0,0)"`). Supports outlier detection (AO/LS/SC), stepwise model selection, ARMA irregular components. Carries an `MSOE`-class object whose `hidden$MSOE = FALSE` / `hidden$PTSnames = TRUE` flags steer the C++ engine into PTS-naming mode.

### R ↔ C++ boundary

There is exactly one C++ entry point exposed to R, declared in `src/RcppExports.cpp` and mirrored in `R/RcppExports.R`:

- `UCompC(commands, ys, us, models, hs, lambdas, outliers, tTests, criterions, periodss, rhoss, verboses, stepwises, p0s, armas, TVPs, seass, trendOptionss, seasonalOptionss, irregularOptionss)` — defined in `src/musecpp2R.cpp`. The `commands` string ("validate", "filter", "smooth", "disturb", "components", "all") selects what the engine does on top of the always-performed estimate + forecast.

The R wrapper carries a large `list` of inputs and outputs (the `MSOE` / `PTS` object), passes inputs to C++, and copies results back into named slots. When adding a new field, update both the R-side `setup` function (initialising to `NA`) and the C++ unpacking/repacking code in `musecpp2R.cpp`.

### C++ implementation layout (`src/*.h`)

Header-only, included from `musecpp2R.cpp`, built around a shared state-space engine:

- `SSpace.h` — generic Kalman-filter/smoother state-space class. The base for all model types.
- `PTSmodel.h` (~3700 lines) — `BSMclass` PTS/BSM model deriving from `SSpace`; this is the engine PTS sits on.
- `ARMAmodel.h` — ARMA model used for the irregular component.
- `ARIMASSmodel.h` — currently `#include`d by `PTSmodel.h` but unused (dead code, flagged for a future cleanup).
- `boxcox.h` — Box-Cox transform / inverse and lambda estimation.
- `optim.h` — numerical optimiser (BFGS-style) used for ML estimation.
- `stats.h`, `DJPTtools.h` — statistical helpers and shared utilities.

When changing model behaviour, the math typically lives in the model-specific header; the R-facing function only edits the input list.

### Documentation generation

`man/` is generated from roxygen comments — never edit `.Rd` files by hand. Shared roxygen fragments live in `man-roxygen/` (e.g. `authors.R`, `keywords.R`) and are included via `@template authors`. `DESCRIPTION` sets `Roxygen: list(old_usage = TRUE)`, so generated usage sections use the legacy style — don't switch it.
