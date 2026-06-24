# muse Package Architecture

## Mental model (read this first)

`muse` has one user-facing function — `pts()` in `R/pts.R` — and one C++ entry point — `UCompC()` in `src/musecpp2R.cpp`.  Every task (estimation, selection, forecasting, simulation, diagnostics) is driven by calling `UCompC()` with a different `command` string (`"all"`, `"forecastOnly"`, `"simulate"`, …).  `pts()` delegates immediately to `.pts_fit()` (`R/pts-internals.R`), which translates the user's 3-letter model string to a UC string via `pts_to_uc()` (`R/pts-translate.R`), marshals arguments into a flat list, and calls `UCompC()`.  The C++ side (`src/musecore.h` → `src/PTSmodel.h` → `src/SSpace.h`) runs the Kalman filter, quasi-Newton optimiser, and component extractor, then returns a named list which `.pts_fit()` post-processes into the `pts` object that `pts()` returns.  All in-sample residuals, fitted values in `$comp`, and innovations live on the **Box-Cox scale**; only `$fitted`, `$forecast`, and accuracy comparisons are back-transformed to the original scale via `.inv_box_cox()`.  One or more variance parameters may be **concentrated out** analytically (not via the optimiser); these appear in `$B` but have `NaN` rows/columns in `$vcov` — `$nParam` still counts them correctly for information criteria.

**Box-Cox λ is chosen on the R side before the engine runs.**  When `model`'s power position is `Z`, `pts()` picks λ via a fast variance-stabilisation screen — classical decomposition (`smooth::msdecompose`, `smoother = "ma"`) followed by Guerrero coefficient-of-variation minimisation over the clipped range `[0, 2]` (`.pts_guerrero_decomp_lambda` in `R/pts-internals.R`) — then rewrites the spec with that numeric λ.  The engine therefore receives a fixed λ in the normal flow; its internal joint-λ BFGS and anchor-snap path (`{−2,−1,−0.5,0,0.5,1,2}`) survive only as a fallback for series too short to screen (`length(y) < 4`).  λ counts as one degree of freedom whenever the screen ran (`lambdaWasScreened`) or the engine estimated it (`lambdaEstimated`); `pts()` adds that DoF on the R side.  The default information criterion is **BICc**.

S3 dispatch for `plot`, `AIC`, `BIC`, and several greybox generics falls through to the `"smooth"` tail of the class vector — no local implementations are needed for those.

`muse` ships **two front-ends over one C++ engine**: this R flow, and a Python package (`python/`) whose `PTS` class binds the identical engine through `pybind11`.  This document describes the R side throughout; the Python counterpart is summarised under "Two front-ends, one engine" (below) and "Python front-end (`python/`)" (near the end), and the R↔Python module mapping is given there.

---

## Package layout

```
muse/
├── R/
│   ├── pts.R                      # User entry point: pts()
│   ├── pts-internals.R            # Core fitting helpers + λ screen (not exported)
│   ├── pts-translate.R            # PTS ↔ UC string translation + selection (not exported)
│   ├── methods.R                  # print / fitted / residuals / coef / forecast / ...
│   ├── pts-summary.R              # summary.pts with coefficient table + proportions
│   ├── pts-methods-accessors.R    # sigma / nparam / actuals / modelType / orders / ...
│   ├── pts-methods-diagnostics.R  # rstandard / rstudent / pointLik / outlierdummy
│   ├── pts-accuracy.R             # accuracy.pts
│   ├── pts-simulate.R             # simulate.pts
│   ├── pts-confint.R              # confint.pts (Wald intervals)
│   ├── pts-update.R               # update.pts
│   ├── pts-pls.R                  # profile log-likelihood utilities
│   ├── muse-package.R            # Package-level roxygen / @useDynLib
│   ├── zzz.R                      # .onLoad / .onAttach + greybox imports
│   └── RcppExports.R              # Auto-generated: declares UCompC() for R
├── src/                          # shared C++ engine (compiled by both fronts)
│   ├── musecpp2R.cpp              # Rcpp bridge: UCompC() → runMuseCommand()
│   ├── musecore.h                 # Language-agnostic dispatch: MuseInputs/Outputs,
│   │                              #   runMuseCommand() + runArmaScore() ← SEXP-free seam
│   ├── muse_compat.h              # Rprintf shim under -DMUSE_PYTHON_BUILD (no-op in R)
│   ├── PTSmodel.h                 # BSMclass (the PTS engine): ~3700 lines
│   ├── SSpace.h                   # SSmodel base: Kalman filter / smoother
│   ├── ARMAmodel.h                # ARMA irregular component
│   ├── boxcox.h                   # BoxCox / invBoxCox / testBoxCox
│   ├── bcnorm.h                   # bcnormBoxCox / bcnormLogJac (BC-normal density)
│   ├── optim.h                    # Quasi-Newton optimiser (BFGS-style)
│   ├── stats.h                    # Statistical helpers (infoCriteria, …)
│   ├── DJPTtools.h                # Shared utilities
│   └── python/
│       └── musecpp2py.cpp         # pybind11 bridge (Python's mirror of musecpp2R.cpp)
├── man/                           # Generated by roxygen2 — do not edit
└── python/                        # Python front-end (build-ignored by R CMD build)
    ├── pyproject.toml             # scikit-build-core build + ruff/mypy/pytest config
    ├── CMakeLists.txt             # pybind11_add_module(_musecore) over ../src
    ├── README.md                  # PyPI landing page
    ├── PYTHON_PORT.md             # port roadmap / phase log
    ├── build_spike.sh             # direct g++ build (dev; bypasses CMake)
    ├── src/muse/
    │   ├── __init__.py            # exports PTS; imports the built _musecore
    │   ├── _musecore*.so          # compiled pybind11 module (UCompC + UCompARMAC)
    │   └── core/
    │       ├── pts.py             # PTS class: fit/predict/properties/summary/plot/...
    │       ├── translate.py       # pts_to_uc / uc_to_pts / orders_to_uc / arma_spec
    │       ├── lambda_screen.py   # guerrero_decomp_lambda (reuses smooth.msdecompose)
    │       ├── selector.py        # select_pts_arma (PTS-then-ARMA order search)
    │       ├── forecaster.py      # forecast() + ForecastResult (interval logic)
    │       ├── boxcox.py          # inv_box_cox
    │       └── io.py              # pandas-Series input + lag inference + index wrap
    └── tests/
        ├── test_functionality.py  # invariant tests (no R) — run in CI
        ├── test_*_parity.py       # R↔Python numeric parity (need R; NOT run in CI)
        └── dump_*_reference.R      # R-side reference dumpers for the parity tests
```

### Two front-ends, one engine

The C++ in `src/` (everything from `musecore.h` down) is **front-end agnostic** —
it speaks only Armadillo + STL and exposes two entry points, `runMuseCommand()`
(the `command`-dispatched estimator/forecaster/simulator) and `runArmaScore()`
(the standalone ARMA scorer used by order selection).  Two thin binding files
wrap those for the two languages:

| | R front-end | Python front-end |
|---|---|---|
| Binding (the only language-specific C++) | `src/musecpp2R.cpp` (`Rcpp`, `SEXP`) | `src/python/musecpp2py.cpp` (`pybind11`, numpy) |
| Exported entry points | `.UCompC`, `.UCompARMAC` | `_musecore.ucomp`, `_musecore.ucomp_arma` |
| User API | `pts()` + S3 methods | `PTS` class (sklearn-style) |
| Build | `R CMD INSTALL` via `src/Makevars` | `pip install` via scikit-build-core + CMake |

The engine's only R-C-API dependency is `Rprintf`; `src/muse_compat.h` supplies a
stdout shim when compiled with `-DMUSE_PYTHON_BUILD`, so the Python build needs no
R.  `R CMD build` ignores `python/` and `src/python/` (see `.Rbuildignore`), and
R only compiles `src/*.cpp` (never the `src/python/` subdir), so the two builds
never collide.  The Python API mirrors `pts()` numerically — every deterministic
output matches R to ~1e-6 or better (parity suites in `python/tests/`).  Full
detail of the port lives in `python/PYTHON_PORT.md`.

---

## The pts object

`pts()` returns an R list of class `c("pts", "smooth")`.  The `"smooth"` tail lets
`plot.smooth.forecast` and several greybox generics dispatch without needing their
own implementations here.  `plot.pts` is a thin pre-processor for panel 12 (state
decomposition) that prepends an `actuals` column to `$states` and then delegates to
`plot.smooth` via `NextMethod()`; all other panels (1-11, 13-14) pass straight through.

Key slots:

| Slot | Type | Scale | Description |
|------|------|-------|-------------|
| `data` | ts/zoo | original | In-sample response |
| `u` | matrix or NULL | — | Regressor matrix (k × n), NULL when unused |
| `model` | string | — | PTS label, e.g. `"PTS(1,G,T)"` |
| `modelUC` | string | — | UC spec, e.g. `"td/none/equal/arma(0,0)"` |
| `lags` | int | — | Fundamental seasonal period |
| `lagsAll` | numeric vec | — | All harmonic periods (lags/1, lags/2, …) |
| `lambda` | double | — | Box-Cox λ (1 = no transform) |
| `B` | named vec | — | Estimated parameter vector |
| `vcov` | matrix | — | Parameter covariance (from Hessian) |
| `nParam` | matrix | — | adam-style 2×5 DoF table (rows `Estimated`/`Provided` × cols `nParamInternal`/`nParamXreg`/`nParamOccurrence`/`nParamScale`/`nParamAll`); `nparam()` reads `[Estimated, nParamAll]`. Initials fold into `nParamInternal` |
| `fitted` | ts/zoo | original | In-sample fitted values |
| `residuals` | ts/zoo | BC | Innovations (BC scale, white-noise sequence) |
| `comp` | matrix | BC | Additive decomposition: Error, Fit, Level, Slope?, Seasonal?, Irregular? |
| `states` | matrix | BC | State evolution (nobs+1 rows, anchored at t=0) |
| `forecast` | ts/zoo | original | Cached h-step forecast (or NA placeholder) |
| `logLik` | double | — | Log-likelihood (BC-corrected) |
| `lossValue` | double | — | -logLik (adam-compatible CFValue) |
| `scale` | double | — | MLE σ on the BC scale |
| `cppOutput` | string | — | Raw C++ validation table |
| `distribution` | string | — | `"dnorm"` (fixed; enables smooth/adam dispatch) |
| `loss` | string | — | `"likelihood"` |
| `holdout` | ts/zoo or NULL | original | Withheld observations when holdout=TRUE |
| `outliers` | string | — | adam-aligned: `"ignore"` or `"use"` |
| `level` | double | — | Confidence level for the outlier z-threshold (default 0.99) |
| `outliersDetected` | data.frame | — | Rows = detected events, columns `time` (1-based) and `type` (factor: AO / LS / SC).  Empty (zero-row) frame when none were found or `outliers = "ignore"`. |

---

## Model strings

### PTS 3-letter spec (`model` argument)

Position 1 — **Power** (Box-Cox λ):
- `Z` — estimate λ via profile likelihood
- A numeric value such as `"0"`, `"0.5"`, `"1"` — fix λ to that value

Position 2 — **Trend**:
- `Z` — auto-select
- `N` — none (random walk, level only)
- `L` — local linear trend (level + slope)
- `D` — damped trend (level + damped slope)
- `G` — global (deterministic) trend

Position 3 — **Seasonal**:
- `Z` — auto-select
- `N` — no seasonal component
- `D` — discrete (linear / harmonic selection)
- `T` — trigonometric (equal variance across all harmonics)

### UC string (internal, passed to C++)

Format: `"<trend>/<cycle>/<seasonal>/<irregular>"`

- trend: `rw` | `llt` | `srw` | `td` | `?`
- cycle: `none` (PTS never uses cycles)
- seasonal: `none` | `linear` | `equal` | `?`
- irregular: `arma(p,q)` | `?`

Translation is handled by `pts_to_uc()` and `uc_to_pts()` in `R/pts-translate.R`.

---

## Task 1: Model estimation (fixed spec)

Call: `pts(data, model = "1LT")`

```
pts()                                         R/pts.R
 │
 ├─ .pts_parse_data(data, formula)            R/pts-internals.R
 │    Extracts y (response) and u (regressors, k×n or NULL)
 │
 ├─ [holdout split if holdout=TRUE && h>0]
 │
 └─ .pts_fit(y, u, model, lags, h, ...)       R/pts-internals.R
      │
      ├─ pts_to_uc(model)                     R/pts-translate.R
      │    Converts "1LT" → modelUC="llt/none/equal/arma(0,0)", lambda=1
      │
      ├─ .pts_uc_inputs(...)
      │    Builds the 21-field list for UCompC:
      │      periods = lags / (1 : floor(lags/2))
      │      rhos    = rep(1, length(periods))
      │      p       = -9999.9   (sentinel → engine initialises parameters)
      │
      ├─ .pts_call_uc("all", args)
      │    Calls UCompC("all", y, u, model, h, lambda, ...)
      │    ─────────────────────── C++ boundary ───────────────────────
      │    musecpp2R.cpp: UCompC()
      │      marshals SEXP → MuseInputs
      │      runMuseCommand(in, out)
      │        preProcess()      — validate dimensions, fix seas/periods
      │        BSMclass object created from SSinputs + BSMmodel
      │        BSMclass::estim() — quasi-Newton on logLik
      │          quasiNewtonBSM()
      │            llik(p) called each iteration:
      │              bsmMatrices(p) — build T, Z, R, H, Q from params
      │              SSmodel::filter()  — Kalman filter pass
      │              return -logLik (including BC Jacobian)
      │          concentrated variances recovered post-optimisation
      │        SSmodel::filter()  — final filter with optimal p
      │        SSmodel::smooth()  — RTS smoother
      │        BSMclass::components() — extract comp matrix (m×n)
      │        BSMclass::validate()   — covp from Hessian, diagnostics table
      │    packs MuseOutputs → Rcpp::List
      │    ─────────────────────── R boundary ────────────────────────
      │
      ├─ extract p (named), covp, lambdaEstimated from out
      │
      ├─ .pts_wrap_oos(out$yFor, y)     ts/zoo-wrap h-step forecast
      ├─ .pts_ts_innov(out$v, y)        ts/zoo-wrap innovations
      ├─ .pts_ts_comp(out$comp, m, y)   reshape m×n → n×m ts/zoo matrix
      ├─ .pts_build_comp(rawComp, v)    add Error, Fit columns; reorder
      ├─ .inv_box_cox(fittedBC, lambda) back-transform fitted values
      │
      └─ return list(modelUC, lambda, p, covp, yFor, comp, fitted,
                     residuals, scale, logLik, lambdaEstimated, nInitial, IC)

 pts() assembles the return object:
   states  = .pts_wrap_states(rbind(NA, comp[, structural cols]))
   ordersList = uc_to_arma(modelUC)
   out$B   = res$p
   # adam-style 2x5 table; .pts_nparam_table() folds initials into Internal,
   # peels one optimised param into Scale, regressors into Xreg.  Total
   # (Estimated, nParamAll) = length(p) + nInitial + lambdaDoF + nXreg.
   out$nParam = .pts_nparam_table(nP = length(p), nInitial = res$nInitial,
                  lambdaDoF = as.integer(lambdaEstimated || lambdaWasScreened),
                  nXreg = if (is.null(u)) 0L else nrow(u))
```

---

## Task 2: Model selection (ZZZ / auto)

Call: `pts(data, model = "ZZZ")`

Selection is split across three stages that run in this order:

**Stage A — λ screen (R side, always for `Z` power).**  Before anything else,
`pts()` resolves the Box-Cox λ via `.pts_guerrero_decomp_lambda()`
(decomposition + Guerrero CV over `[0, 2]`; see "Task 1b: λ screen" below) and
rewrites the spec from `Z??` to `<num>??`.  No state-space model is fitted here.

**Stage B — structural trend/seasonal ident (C++ engine).**  The remaining `Z`
tokens in the trend/seasonal positions become `?` in the UC string, and the engine
searches over them:

```
runMuseCommand("all", ...)
  BSMclass::ident()                  triggered by '?' tokens in model string
    findUCmodels() + estimUCs()
      Loops over candidate combinations:
        (trend ∈ {rw, llt, srw, td}) × (seasonal ∈ {none, linear, equal})
        (the (damped trend, AR>0) combinations are filtered out)
      For each candidate: estim() at the fixed λ, score by criterion
      Selects best candidate by chosen criterion (default BICc)
    [optional] selectHarmonics() — if seasonal='?' drops unused harmonics
  Continues with estim() / filter() / smooth() on the winner
```

Because λ is already numeric by this point, the engine's joint-λ BFGS / anchor-snap
machinery is **not exercised** in the normal flow — it remains only for the
short-series fallback noted in the mental model.

**Stage C — ARMA order selection (R side, only when `orders$select = TRUE`).**
When the user requests ARMA search, `pts()` calls `.pts_select_pts_arma()`
(`R/pts-translate.R`) *instead of* relying on a nested engine search.  It is a
sequential two-pass strategy:

```
.pts_select_pts_arma()                         R/pts-translate.R
  Pass 1: fit every PTS structural shape (no ARMA) once, rank by IC
          (k counts λ when screened, +1 for the G/td drift slope)
  Pass 2: cascaded ARMA on the winning structural's residuals —
          peel highest-period seasonal lag → … → non-seasonal (lag 1),
          each rung a small (p,q) grid scored by .UCompARMAC, the (0,0)
          cell always present so "no ARMA at this lag" is a valid choice
  Returns the locked (model_spec, ar, ma, lags); the final .pts_fit
  then runs at fixed structure + fixed ARMA (armaIdent = FALSE).
```

A naive nested loop (ARMA grid inside the structural loop) would re-fit the full
ARMA cap up to N_structural times; the sequential split fits each structural shape
once and runs the ARMA grid only on the winner.

After selection the rest of the R-side post-processing is identical to Task 1.
The verbose ident table printed during `pts(..., verbose=TRUE)` is assembled inside
`estimUCs()` (Stage B) and the R-side grid printers (Stage C); the C++ block is
returned in `out$table` → stored in `object$cppOutput`.

---

## Task 1b: λ screen (decomposition + Guerrero)

Call path: `pts()` → `.pts_guerrero_decomp_lambda()` (`R/pts-internals.R`), invoked
whenever the power position is `Z` and `length(y) >= 4`.

```
.pts_guerrero_decomp_lambda(y, lags, lambda_lower=0, lambda_upper=2)
  guard: y all finite & > 0; m = tail(lags,1) >= 2; n >= 2m   (else return 1)
  decomp = smooth::msdecompose(y, lags = m, type = "additive", smoother = "ma")
  mu_t   = decomp$states[, 1]                  # smoothed trend (level)
  blocks of length m (R = floor(n/m)):
     mu_b[i] = mean(mu_t in block i)
     sd_b[i] = sd((y - mu_t) in block i)       # keeps the seasonal swing in
  lambda = argmin_λ∈[0,2]  CV( sd_b * mu_b^(λ-1) )      via stats::optimize
```

Rationale and the deliberate choice to keep the seasonal swing in `sd_b` (so
multiplicative seasonality is detectable) are documented inline at the call site in
`R/pts.R` and in `NEWS`.  The `[0, 2]` clip removes the inverse-BC `-1/λ` asymptote
that produced `Inf` forecasts under the previous Brent-on-LT-AIC screen.

---

## Task 3: Forecasting

Call: `forecast(object, h = 24)`

```
forecast.pts()                                R/methods.R
 │
 ├─ .pts_forecast_inputs(object, h, newdata) R/pts-internals.R
 │    Rebuilds the UCompC argument list from the fitted object.
 │    Key difference from estimation:
 │      p = object$B   (fixed estimated parameters; no re-estimation)
 │      u: if regressors, concatenates object$u (k×n) with t(newdata)[,1:h] (k×h)
 │
 ├─ .pts_call_uc("forecastOnly", args)
 │    ─────────────────────── C++ boundary ───────────────────────
 │    runMuseCommand("forecastOnly", ...)
 │      BSMclass::setEstimatedParams(p)
 │        bsmMatrices(p) — rebuild T, Z, R, H, Q
 │        SSmodel::filter() — run filter over training data to recover
 │                            terminal state a_n, P_n
 │      SSmodel::forecast() — propagate h steps forward:
 │        a_{n+k} = T * a_{n+k-1}  (+ Gam * u_{n+k} if xreg)
 │        P_{n+k} = T * P_{n+k-1} * T' + R * Q * R'
 │        yFor[k] = Z * a_{n+k}
 │        FFor[k] = Z * P_{n+k} * Z' + CHCt    (prediction variance)
 │    returns yFor (BC scale, length h), yForV (prediction variance)
 │    ─────────────────────── R boundary ────────────────────────
 │
 ├─ .pts_wrap_oos(out$yFor, object$data)  attach time index
 ├─ .inv_box_cox(yFor_bc, lambda)         back-transform mean forecast
 │
 ├─ Branch on `interval`:
 │    "prediction" (default)  use yForV
 │    "confidence"            use yForVconf = max(0, yForV - sigma^2_BC)
 │                            (var(E[y|obs]) = var(y|obs) - sigma_obs^2;
 │                            sigma^2 read off object$scale^2 — no reforecast)
 │    "simulated"             empirical quantiles of simulate.pts() paths
 │    "none"                  lower = upper = mean
 │  Vector `level` -> lower/upper become (h × nLevels) matrices.
 │  `side` (both / upper / lower) derives (qLow, qUp); qnorm(0) = -Inf
 │  flows through .inv_box_cox to the BC support boundary (0 for
 │  lambda > 0, -Inf for the identity transform).
 │  If `cumulative = TRUE`: collapse to one row via simulation totals
 │  (exact for "simulated"; approximation otherwise because the engine
 │  does not expose cross-step state covariance).
 │
 └─ return list of class c("pts.forecast", "smooth.forecast")
      mean, lower, upper (original scale), variance (BC scale),
      level, interval, side, cumulative,
      scenarios (only when interval = "simulated" and scenarios = TRUE)
```

`predict.pts()` is an alias for `fitted.pts()` — in-sample fitted values only.

---

## Task 4: Simulation

Call: `simulate(object, nsim = 100, h = 24)`

```
simulate.pts()                                R/pts-simulate.R
 │
 ├─ Save/restore R RNG state (C++ uses R's randn when linked)
 │
 ├─ .pts_forecast_inputs(object, h)
 │    Same as forecasting (p fixed, u extended if xreg)
 │    args$nsim = nsim, args$seed = seed
 │
 ├─ .pts_call_uc("simulate", args)
 │    ─────────────────────── C++ boundary ───────────────────────
 │    runMuseCommand("simulate", ...)
 │      BSMclass::setEstimatedParams(p)
 │      SSmodel::simulate(nsim, seed)
 │        for each path i = 1..nsim:
 │          draw eta_t ~ N(0, Q) (state disturbances)
 │          draw eps_t ~ N(0, H) (observation noise)
 │          propagate state forward h steps
 │          record yFor_i[1..h]
 │        returns simPaths (h × nsim, BC scale)
 │    ─────────────────────── R boundary ────────────────────────
 │
 ├─ .pts_wrap_oos() + .inv_box_cox() per column
 │
 └─ return list of class c("pts.sim", "smooth.sim")
      data (h × nsim matrix, original scale)
```

---

## Task 5: Summary and coefficient table

Call: `summary(object)`

```
summary.pts()                                 R/pts-summary.R
 │
 ├─ est = coef(object)        # object$B, full parameter vector
 ├─ cv  = vcov(object)        # object$vcov, from Hessian
 │
 ├─ Wald t-statistics and CI for every non-Irregular parameter:
 │    ses   = sqrt(diag(cv))
 │    tval  = est / ses
 │    pval  = 2*(1 - pnorm(|tval|))
 │    lower/upper = est ± qnorm(1-α/2) * ses
 │    Irregular excluded from the table (task 3)
 │
 ├─ Variance proportions (structural params only, no Damping/Irregular/ARMA/Beta):
 │    props  = varVals / sum(varVals)
 │    Delta-method SEs:
 │      J[i,j] = dp_i/dv_j = { (S-v_i)/S² if i==j,  -v_i/S² otherwise }
 │      propVar = diag(J %*% Sv %*% t(J))   where Sv = vcov submatrix
 │
 └─ return summary.pts object with $coefficients, $proportions, $IC, ...

print.summary.pts()
 │
 ├─ Header: model, UC string, lambda, periods, nobs, nParam, sigma
 ├─ Coefficient table (Estimate, SE, t, p, Lower, Upper)
 ├─ Variance proportions table (Proportion, SE)
 └─ Log-likelihood and information criteria
```

---

## Task 6: Diagnostics

### Residual diagnostics

```
rstandard.pts(model)      → residuals(model) / sigma(model)
rstudent.pts(model)       → same (conservative approximation for state-space)
pointLik.pts(object)      → dnorm(residuals, sd=sigma, log=TRUE)
outlierdummy.pts(object, level, type)
   computes |rstandard| > qnorm((1+level)/2) and returns index + bound
```

### Accuracy (holdout evaluation)

```
accuracy.pts(object, holdout)             R/pts-accuracy.R
 │
 ├─ holdout = object$holdout  if not supplied
 ├─ forecast(object, h=length(holdout))  → calls forecast.pts()
 └─ greybox::measures(holdout, forecast$mean)  → MAE, MASE, RMSE, ...
```

### Confidence intervals

```
confint.pts(object, parm, level)          R/pts-confint.R
 │
 ├─ est = coef(object)
 ├─ ses = sqrt(diag(vcov(object)))
 ├─ z   = qnorm(1 - (1-level)/2)
 └─ return [est - z*ses, est + z*ses]    (Wald, warns on NaN for concentrated params)
```

---

## Task 7: Outlier detection (`outliers = "use"`)

Call: `pts(data, model = "PTS(Z,Z,Z)", outliers = "use", level = 0.99)`

User-facing trichotomy mirrors `smooth::adam()`:

* `"ignore"` (default) — no outlier handling.
* `"use"` — run the engine's outlier detector once, classify each
  detected event as AO / LS / SC, and refit with the detected events
  appended as fixed regressor dummies.
* `"select"` — not yet supported; the R side rejects this value with a
  clear message.

`level` (default 0.99) is converted to a positive z-score threshold via
`stats::qnorm((1 + level) / 2)` and passed through to the C++ engine
as the `outlier` field of `SSinputs`.  Inside `BSMclass::estimOutlier()`
the AO threshold equals the user z (in absolute value — the engine
flags the "final fit with detection" pass via a negative outlier
value, so the C++ side uses `std::abs(inputs.outlier)` to recover the
user-supplied threshold); LS and SC use the same z scaled by the
engine's original relative stiffness (LS = z × 2.5/2.3, SC = z ×
3.0/2.3).  After detection the engine writes the (type, time) rows into
`BSMmodel::typeOutliers` (engine codes: 0 = AO, 1 = LS, 2 = SC).

**Auxiliary residuals (`auxFilter`, `smooth == 2`).**  The AO statistic is the
standardised smoothed *observation* disturbance; LS / SC use the level / slope
disturbances.  Each component's residual is standardised by its **own empirical
SD** (per column), NOT via a single matrix `pinv()` across all components — a
shared `pinv()` tolerance would treat any component whose variance is far below
the largest (e.g. a near-zero observation variance) as singular and zero it,
destroying the outlier signal exactly where it matters.  The per-step
disturbance is left raw (`Q R' r_t`, no division by the smoothed-disturbance
variance `Veta`, which is singular when an estimated variance collapses).

**Acceptance.**  `estimOutlier` keeps an outlier through backward deletion only
if its augmented-KF coefficient t-stat clears the `level` z-threshold, and then
keeps the whole outlier model unless it failed to converge.  It does **not**
revert on "IC worse than the no-outlier baseline": an over-fit baseline (the
unbounded variance→0 likelihood singularity on short, flexible models) has an
artificially good IC that no genuine outlier model can beat, which used to drop
real, highly-significant outliers.

### Joint-λ + outlier-injection workaround

The engine's outlier-injection refit path has a parameter-vector
dimensionality bug when joint-BFGS is also estimating λ — the extra λ
slot does not survive the dummy-injection refit and the BFGS aborts
with a `4x1 vs 3x1` dimension mismatch.

`pts()` sidesteps this on the R side: when `outliers = "use"` *and* the
caller's model spec has `Z` in position 1 (auto-λ), the function runs a
quick **no-outlier** preliminary fit, reads the resulting λ, rounds it
to four decimals, and rewrites the model spec with that numeric value
before invoking the outlier-detecting fit.  The preliminary fit is
cheap (≈ 50 ms) and the rounding has no measurable effect on the final
likelihood.

### R-side post-processing

```
pts() with outliers = "use"
 │
 ├─ [optional λ-pinning preliminary fit — see above]
 │
 ├─ .pts_fit(..., outlier = qnorm((1 + level) / 2))
 │    Engine returns out$typeOutliers (n × 2 matrix; empty when none)
 │
 ├─ .pts_fit() converts to data.frame:
 │    data.frame(time = typeOutliers[, 2],            # 1-based
 │               type = factor(c("AO", "LS", "SC")[typeOutliers[, 1] + 1L],
 │                             levels = c("AO", "LS", "SC")))
 │
 └─ pts() stores it as $outliersDetected on the return list
```

Detected dummies enter the standard parameter machinery as `AO<t>`,
`LS<t>`, `SC<t>` rows: they appear in `coef(m)`, in `summary(m)`'s
"Outlier coefficients" block, and on the `print(m)` "X type outliers
detected" line.  They are filtered out of the variance-proportions
table by a `grepl("^(AO|LS|SC)[0-9]+$", nm)` guard shared between
`print.pts` and `summary.pts`.

`vcov(m)` only covers structural parameters (the Hessian is computed in
ratio space before the dummies are injected), so the outlier-dummy
rows in `coef(m)` get `NA` standard errors.  Information criteria
(`AIC`, `BIC`, …) count the dummies correctly via `length(B) +
lambdaEstimated`.

**Covariance estimator (`BSMclass::parCov`, `PTSmodel.h`).**  The parameter
covariance is built from the observed-information Hessian, evaluated on the
*absolute* (non-concentrated, `bsmMatricesTrue`) scale at the optimum.  The
Hessian (`hessLlik`, `SSpace.h`) uses **central second differences** with a
per-parameter step `eps^(1/4)·max(|p|,1)` (not one-sided forward differences
with a fixed step — that was biased and cancelled catastrophically in flat
directions).  If the resulting Hessian submatrix is indefinite or numerically
singular (`min eig ≤ 1e-8·max eig` — a boundary / weakly-identified variance,
or an ARMA φ≈θ near-cancellation), `parCov` falls back to the **OPG / BHHH**
estimator `Σ_t s_t s_tᵀ`, where `s_t = ∂/∂θ [0.5(log F_t + v_t²/F_t) −
logJac_t]` are per-observation scores (central-differenced from the stored
`inputs.v`/`inputs.F`).  OPG is PSD by construction, so the returned `vcov` is
always a valid covariance.  **Caveat:** the augmented (xreg) path concentrates
its regression coefficients in the augmented filter and stores no
per-observation innovations, so OPG is unavailable there — augmented models
keep the (improved) Hessian.

---

## The C++ boundary in detail

### UCompC() input list (`command` + 22 args)

The full positional signature is `UCompC(command, y, u, model, h, lambda, outlier,
tTest, criterion, periods, rhos, verbose, stepwise, p, arma, TVP, seas,
trendOptions, seasonalOptions, irregularOptions, nsim, seed, lambdaLower)`.

| Field | R type | C++ field | Notes |
|-------|--------|-----------|-------|
| `y` | numeric vec | `SSinputs.y` | Response (may be BC-transformed by engine) |
| `u` | numeric matrix | `SSinputs.u` | k×n regressors; `matrix(0,1,2)` sentinel when unused |
| `model` | string | `BSMmodel.model` | UC string with possible `?` tokens |
| `h` | int | `SSinputs.h` | Forecast horizon |
| `lambda` | double | `BSMmodel.lambda` | 9999.9 → estimate; else fix |
| `outlier` | double | `BSMmodel.outlier` | 0 = none, >0 = threshold |
| `tTest` | bool | `BSMmodel.tTest` | t-test based component selection |
| `criterion` | string | `BSMmodel.criterion` | `"aic"` / `"aicc"` / `"bic"` / `"bicc"` |
| `periods` | numeric vec | `SSinputs.periods` | Harmonic periods |
| `rhos` | numeric vec | `SSinputs.rhos` | Rho flags for each harmonic |
| `verbose` | bool | `SSinputs.verbose` | Print ident table |
| `stepwise` | bool | `BSMmodel.stepwise` | Stepwise ARMA selection |
| `p` | numeric vec | `SSinputs.p` | Initial / fixed parameters (`-9999.9` → engine chooses) |
| `arma` | bool | `BSMmodel.arma` | Search ARMA orders |
| `TVP` | double | — | Time-varying parameters flag (`-9999.99` = off) |
| `seas` | double | `BSMmodel.seas` | Fundamental seasonal period |
| `trendOptions` | string | — | `"rw/llt/srw/td"` |
| `seasonalOptions` | string | — | `"none/linear/equal"` |
| `irregularOptions` | string | — | `"arma(0,0)"` |
| `nsim` | int | `MuseInputs.nsim` | Simulation paths (1 = off) |
| `seed` | int | `MuseInputs.seed` | RNG seed for simulate |
| `lambdaLower` | double | `SSinputs.lambdaLower` | Lower bound for engine-side λ (`1e-10` when data has zeros; `-Inf` = unbounded) |

### UCompC() output list (command-dependent)

| Field | Commands | Description |
|-------|----------|-------------|
| `model` | all | Resolved UC string |
| `yFor` | all | BC-scale forecast (length h) |
| `yForV` | all | BC-scale prediction variance: Z·Pt·Zᵀ + CHCt with Pt evolving with full RQRᵀ injection (length h).  R-side confidence variance is computed as `max(0, yForV - sigma^2)` -- no separate engine field. |
| `lambda` | all | Final Box-Cox λ |
| `lambdaEstimated` | all | TRUE if λ cost a DoF |
| `p` | all | Final parameter vector |
| `p0` | all | Initial parameter vector |
| `parNames` | all | Parameter names |
| `criteria` | all | logLik, AIC, BIC, AICc (length 4) |
| `coef` | all, validate | Alias for p (used for named extraction) |
| `covp` | all, validate | Parameter covariance matrix |
| `table` | all, validate | Diagnostics text block |
| `typeOutliers` | all, validate | (n × 2) matrix of detected outliers; column 1 = engine type code (0 = AO, 1 = LS, 2 = SC), column 2 = 1-based time index.  Zero-row when none were found.  Surfaced to R as `$outliersDetected`. |
| `objFunValue` | all | Final BFGS objective value (BCnorm marginal log-likelihood × −2/n) |
| `v` | all, filter | Innovations (length n) |
| `a` | all, filter | Filtered states |
| `P` | all, filter | Filtered state variances |
| `yFit` | all, filter | Filtered fitted values |
| `yFitV` | all, filter | Filtered fitted variances |
| `eps` | filter | Smoothed observation disturbances |
| `eta` | filter | Smoothed state disturbances |
| `stateNames` | filter | `/`-separated state-component names |
| `comp` | all, components | Component matrix (m×n, column-major) |
| `compV` | all, components | Component variance matrix |
| `m` | all, components | Number of components |
| `compNames` | all, components | `/`-separated component names |
| `simPaths` | simulate | h×nsim simulated paths (original scale) |

---

## Python front-end (`python/`)

The Python package is a thin API layer over the **same** engine: it does no
state-space mathematics of its own.  `python/src/muse/core/pts.py`'s `PTS`
class is the scikit-learn-style analogue of `pts()` — spec in the constructor,
data into `.fit(y)`, results via properties and `.predict()`.  Each R file has
a direct Python counterpart:

| Concern | R | Python |
|---------|---|--------|
| Entry point / object | `pts()` + `pts` S3 object | `PTS` class (`core/pts.py`) |
| Spec ↔ UC translation | `pts_to_uc` / `uc_to_pts` / `uc_to_arma` / `.pts_orders_to_uc` | `core/translate.py` (`pts_to_uc`, `uc_to_pts`, `orders_to_uc`, `arma_spec`) |
| λ screen | `.pts_guerrero_decomp_lambda` | `core/lambda_screen.py` (reuses `smooth.msdecompose`) |
| PTS-then-ARMA selection | `.pts_select_pts_arma` | `core/selector.py` |
| Forecast + intervals | `forecast.pts` | `core/forecaster.py` + `ForecastResult` |
| Box-Cox inverse | `.inv_box_cox` | `core/boxcox.py` |
| ts/zoo wrapping | `.pts_wrap_*` | `core/io.py` (pandas index + lag inference) |
| Engine marshalling | `.pts_uc_inputs` / `.pts_call_uc` | `PTS._engine` / `PTS._forecast_engine` / `PTS._fit_structural` |

### Engine calls

`musecpp2py.cpp` exposes two functions, mirroring the R bridge:

- `_musecore.ucomp(command, y, u, model, h, lambda, outlier, …, lambdaLower)` —
  the same positional argument list as `.UCompC`; numpy arrays are copied to/from
  Armadillo (no `carma`).  `command` is `"all"` (fit), `"forecastOnly"`,
  `"simulate"`, or `"simulateInit"`.  Returns a `dict` keyed exactly like the R
  output list.
- `_musecore.ucomp_arma(y, ar_orders, ma_orders, arma_lags)` — mirrors
  `.UCompARMAC`, calling the shared `runArmaScore()`; used by `selector.py`.

### Differences from R (deliberate)

- **`coef` source.** `PTS` reads natural-scale variances from `out["coef"]`, not
  the optimiser-space `out["p"]` (same as R's `coef()`).
- **`orders` API.** ARMA orders are `ar`/`ma` scalars (non-seasonal) or length-2
  vectors (SARMA); the seasonal lag comes from the top-level `lags`, **not** from
  `orders` (`orders["lags"]` raises).  R additionally allows `orders$lags`.
- **No auto-distribution / formula.** Only `distribution = "dnorm"`; xreg is
  passed as the `X` array (no formula interface).
- **Optional deps.** Auto-λ needs `smooth` (`msdecompose`); `accuracy()` needs
  `greybox` (`measures`); `plot()` reuses `smooth`'s `plot_adam` via a duck-typed
  adapter.  RNG-based paths (`simulate`, simulated intervals) are statistical-only
  parity (R and Python RNG streams differ).

### Tests

- `python/tests/test_functionality.py` — invariant/shape tests, **no R**; the CI
  job (`.github/workflows/python-check.yaml`) runs these plus `ruff` and `mypy`.
- `python/tests/test_*_parity.py` — exhaustive numeric parity against R, fed by
  the `dump_*_reference.R` scripts; **not** run in CI (need R + reference JSON).
  Every deterministic output matches R to ≤1e-6 (most to machine precision).

---

## Parameter naming convention (C++)

Set by `BSMclass::parLabels()` in `PTSmodel.h`:

```
Damping          — if trend is damped (srw/td); typePar = -1
Level            — always present;              typePar =  0  (variance)
Slope            — if trend has slope;          typePar =  0
Seas(All)        — equal seasonal variance;     typePar =  0
Seas(<period>)   — per-harmonic variance;       typePar =  0
Irregular        — irregular variance;          typePar =  0
AR(<i>)          — ARMA AR coefficients;        typePar =  3
MA(<i>)          — ARMA MA coefficients;        typePar =  3
Beta(<i>)        — regression coefficients;     typePar =  4
```

`constPar` encodes how the parameter was handled by the optimiser:
- `0` — free (Hessian-based SE available)
- `1` — concentrated out analytically (SE = NaN in vcov; marked with `*` by task 13)
- `2` — variance fixed to zero
- `3` — other constraint (alpha = 0 or 1)

---

## State-space matrices (SSpace.h)

The Kalman filter runs on the system:

```
Measurement:   y_t = Z * a_t + D * u_t + C * eps_t,   Var(eps_t) = H
Transition:   a_t+1 = T * a_t + Γ * u_t + R * eta_t,  Var(eta_t) = Q
Covariance:   Cov(eta_t, eps_t) = S
```

`BSMclass::bsmMatrices(p)` populates T, Z, R, H, Q from the parameter vector `p`
every time the optimiser proposes a new candidate.  Z and T are structurally fixed
by the model type; only the variance entries of Q (and H) are free parameters.

### Sparsity exploitation (high-lag performance)

T is stored as a dense `arma::mat` but is **~98% zeros by design**: block-diagonal
2×2 rotation blocks per trig harmonic (`[[c,s],[-s,c]]`), companion sub-diagonals
for dummy-seasonal / ARMA.  At high seasonal lags (`m ≈ s`, e.g. 336) the dense
`T·P·Tᵀ` Kalman products are the dominant O(m³) cost.  The hot loops therefore
build a **local sparse view** `arma::sp_mat Tsp(system.T)` once per pass (the O(m²)
conversion is negligible) and use it for every per-timestep product, turning
`T·P·Tᵀ` into O(m²) (`sp_mat · mat → mat`):

- **Forward filter** (`KFprediction`, `llik`/`auxFilter`/`forecast`/`KFinnovations`):
  sparse-`T` overload; `P` stays dense (it fills in during filtering).  Triple
  products are split into explicit binary products to stay on the dense-result path.
- **Analytic gradient** (`gradLlik`): sparse `Tᵀ·Nt·T`, plus a **rank-1** expansion
  of `Lt = I − K·Z` (`Lt'·Nt·Lt = Nt − w z' − z w' + (k'w) z z'`, `w = Nt·k`),
  collapsing the O(m³) backward recursion to O(m²).

**Gradient method & optimiser robustness.**  The model-setup logic
(`PTSmodel.h`, ~`SSmodel::inputs.exact = …`) chooses the gradient method per
model: the **analytic** disturbance-smoother gradient (`exact = true`) for pure
structural models, and the **numerical** gradient (`exact = false`) for ARMA
(AR coefs enter `T`, which the disturbance-smoother formula cannot
differentiate), damped trend, cycle, and regressor (augmented-KF) models.  The
analytic gradient must build its baseline `Q`/`H` at the **same point** the
gradient is evaluated (`gradLlik` calls `userModel(p)` before `sysmatQ`) — the
smoother accumulants are on the **absolute** scale (`iFt` is divided by
`innVariance`), so leaving `Q`/`H` on the stale ratio scale makes `dQ` mix units
and blow up by ~`1/innVariance`.  It also normalises by `nFinite = n − nMiss`
(the same divisor `llik()` averages over), **not** `n − nMiss − d_t − 1`.
Validate any change against a central-difference reference (the two must agree to
~6 digits).  `quasiNewtonBSM` carries a **descent-direction safeguard**: if the
BFGS inverse-Hessian yields `d` with `d·grad ≥ 0` it resets `iHess = I`
(steepest descent) so the line search can always make progress instead of
stalling at a non-stationary point.
- **State smoother** (`auxFilter` under `inputs.stateOnly`, set by `components()`):
  the entire backward `Nt` recursion and the O(m³) `P·Nt·P` smoothed-variance update
  are **skipped** — those outputs (`data.P` → dead `compV`; outlier-mode
  `rNrOut`/`rOut`/`NOut`) are never surfaced.  Smoothed *states* depend only on the
  `rt` recursion, so they are unchanged.  The disturbance/outlier-detection path
  (`smooth == 2/3`, `outliers = "use"`) keeps the full `Nt` recursion.

Sparse-hostile setup ops (`eig_gen` stationarity, `schur`/`dlyap` diffuse init) keep
a dense `T`; they run once per likelihood eval (~2% of cost), not per timestep.
All of this is **numerically transparent**: point estimates are bit-identical, and
`vcov`/`confint` now agree across the dense and sparse paths to floating-point
tolerance.  (Earlier the finite-difference Hessian diverged by tens of percent on
ill-conditioned models, because the one-sided forward differences amplified the
~1e-10 sparse-reordering noise through a near-singular inverse; the central-
difference + OPG covariance — see `parCov` below — removed that sensitivity.)

---

## Key invariants

- **Variance floor.** `bsmMatrices` / `bsmMatricesTrue` clamp the variance
  log-parameters (`typePar == 0`) so `exp(2*p) >= exp(-23) ≈ 1e-10` before
  filling `Q`/`H`.  A variance that underflows to *exactly* 0 (a collapsed /
  near-deterministic component) makes the Kalman innovation variance
  `F_t = Z P Zᵀ + H` zero, the gain `K = P Zᵀ/F_t` becomes `0/0 = NaN`, and the
  filtered/terminal state — hence any forecast — is `NaN`/explosive.  The floor
  is applied only to genuine variance parameters; **structural zeros** (e.g.
  seasonal/ARMA companion-state rows of `Q`) are set after and stay 0.

- **BC scale vs original scale.** Residuals, innovations, and the `comp` matrix are
  all on the Box-Cox scale.  `fitted`, `forecast$mean`, and holdout comparisons are
  on the original scale (back-transformed by `.inv_box_cox()`).  Prediction intervals
  are computed by endpoint-transforming the BC-scale ±z*se bounds — not by
  transforming a symmetric original-scale interval.

- **Box-Cox branch convention.** The R-side `.inv_box_cox()` (`R/pts-internals.R`)
  and the C++ `BoxCox()` / `invBoxCox()` (`src/boxcox.h`) plus
  `bcnormBoxCox()` / `bcnormLogJac()` (`src/bcnorm.h`) all use **exact-equality**
  branches: `λ == 0` → log, `λ == 1` → identity, otherwise the general
  `(y^λ − 1) / λ` formula.  No threshold shortcuts (e.g. `|λ| < 0.02`) —
  thresholds make the AIC discontinuous in λ and bias the profile-λ search
  toward the threshold endpoints.

- **`testBoxCox` NaN sentinels.** Every "decomposition failed" branch inside
  `testBoxCox()` (`src/boxcox.h`) returns a large *negative* value
  (`-1e10`, `-1e20`) so the failed candidate loses every subsequent
  `bestLLIK < cLLIK` comparison.  Earlier versions used *positive* sentinels
  for the log / Box-Cox-aux candidates, which caused failed candidates to
  spuriously win and pinned λ to 0 whenever the input contained NaN entries.
  `testBoxCox` also forward-fills NaN entries in `y` up front (then back-fills
  any leading NaN) so the harmonic-regression decomposition test sees a finite
  series; naive `find_finite` filtering would break seasonal periodicity.

- **Regressor matrix orientation.** The user-facing convention is (n observations ×
  k variables); internally and in C++ it is (k variables × n observations).  The
  transpose happens once in `.pts_parse_data()` and once in `.pts_forecast_inputs()`
  for `newdata`.

- **Component matrix shape.** C++ returns the component matrix as a flat vector of
  length m×(n+h) in column-major order (m components, n+h time points).
  `.pts_ts_comp()` reshapes to (n+h)×m and attaches ts/zoo attributes.

- **Variance parameterisation: ratios inside the optimiser, absolute values in output.**
  `bsmMatrices()` builds Q and H as `exp(2·p_i)` for each variance parameter.  With
  concentrated likelihood (`cLlik = true`, the default), one variance is pinned at
  `p = 0` → `exp(0) = 1` during optimisation, so every other `exp(2·p_i)` is a
  **ratio**: σ²_i / σ²_concentrated.  After optimisation, `parameterValues()`
  (`PTSmodel.h`) multiplies all variance entries by `innVariance` — the analytical
  MLE of the concentrated variance computed from the Kalman filter residuals as
  `Σ(v²_t / F_t) / n_finite` (MLE divisor, NOT REML `n − k`) — converting ratios
  to **absolute variances** on the BC scale.
  This is what ends up in `object$B`.  Consequence: the proportions printed by
  `print.pts` are proportions of absolute variances, not of ratios.  The concentrated
  parameter itself appears in `p` / `object$B` with its recovered absolute value, but
  its row/column in `vcov` is NaN (`constPar = 1`) because the Hessian is computed in
  ratio space.  `nParam` counts all free + concentrated parameters correctly for ICs.

- **Which variance is concentrated out — initial choice and dynamic switching.**
  `initParBsm()` Section 7 sets the initial concentrated parameter: if an irregular
  component is present, the Irregular variance is chosen (index `nPar(0)+nPar(1)+nPar(2)`);
  if there is no irregular component, the first variance parameter (Level) is chosen.
  During every BFGS iteration inside `quasiNewtonBSM()`, the code converts the
  current ratio-space `xNew` back to absolute-scale `xUncon`, then checks whether the
  currently concentrated parameter is still the largest variance.  If not, it switches:
  the old concentrated parameter is freed, the new largest becomes concentrated, and all
  ratios are rescaled so the new concentrated parameter sits at `p = 0`.  Damping is
  always excluded from this competition (its index is masked with −300 before
  `index_max()`; its `typePar` is −1 rather than 0 in any case).  The rationale:
  concentrating out the largest variance keeps all remaining ratios ≤ 1, giving a
  better-conditioned BFGS search space.

- **Lambda DoF.** λ costs one degree of freedom whenever it was chosen from the
  data.  In the normal flow this happens via the R-side screen
  (`lambdaWasScreened = TRUE`); on the short-series fallback the engine's joint-λ
  path sets `lambdaEstimated = TRUE` instead (and clears it when the optimised λ
  snaps to a fixed anchor).  `pts()` adds `as.integer(lambdaEstimated ||
  lambdaWasScreened)`.  A fixed numeric λ in the spec (e.g. `"0.5LT"`) costs no DoF.

- **Estimated diffuse initials counted in k; adam-style `nParam` table.** The
  estimated initial states — level, slope (for `L`/`G`/`D`), cycle, and seasonal
  — are diffuse / profiled out and so are genuine degrees of freedom; charging
  for them stops the IC under-penalising flexible seasonal and trend shapes.
  The initial count is `nInitial = ns(0) + ns(1) + ns(2)` (trend + cycle +
  seasonal state dimensions); the **stationary ARMA block `ns(3)` is excluded**
  (its initial is the stationary distribution, not free — the ARMA *coefficients*
  remain counted via `length(B)`).  Read from the engine's own state dimensions,
  it is **lags-driven** and correct for multi-seasonal `lags = c(m1, m2, …)`
  with no hard-coded period sizes; computed once in `runMuseCommand`
  (`out.nInitial = ns(0)+ns(1)+ns(2)`) and surfaced over both bindings.

  The user-facing `$nParam` is an **adam-style 2×5 matrix** (mirrors
  `smooth::adam`): rows `Estimated`/`Provided` × columns `nParamInternal`,
  `nParamXreg`, `nParamOccurrence`, `nParamScale`, `nParamAll`, built by
  `.pts_nparam_table()` (R) / `_nparam_table()` (Python).  The **initials fold
  into `nParamInternal`** (there is *no* separate `nInitial` slot); one optimised
  parameter is peeled into `nParamScale` (the concentrated variance, loss is
  always `likelihood`); regressors go to `nParamXreg`; `nParamOccurrence` is
  always 0.  `nparam()` returns the `[Estimated, nParamAll]` cell =
  `length(B) + nInitial + λDoF + nXreg`.

  The count is added to k in **four coupled places that must stay in sync**: the
  C++ `kFor` lambda in `BSMclass::estim()` (`+ nInitial + u.n_rows`, which drives
  *engine model selection*), the post-fit `$nParam` table in `pts()` and
  `PTS._post_process`, and the R/Python selector `k_struct`
  (`+ struct$nInitial` / `+ st["n_initial"]`).  `nInitial` **subsumes the old
  G/td drift term**: the deterministic drift *is* the initial slope, already
  inside `ns(0) = 2`, so the former `+ (inputs.Drift ? 1 : 0)` / `+ (tr == "G")`
  / `+ grepl("^td/", modelUC)` corrections were removed.  Note: `coef`, `vcov`,
  and `confint` cover only the estimated coefficients (those with a covariance
  row), so `length(coef) == nParam − nInitial − λ`, not `nParam`.

- **MLE σ̂² and BCnorm consistency.**  Both `llik()` and `llikAug()` in
  `src/SSpace.h` use the **MLE divisor** `n_finite` (= total finite observations)
  for the residual variance estimator — not the REML / `n − k` divisor.  The
  Box-Cox Jacobian (`Σ_t bcnormLogJac(y_raw_t, λ)`) is also summed over the
  same `n_finite` observations and folded directly into `objFunValue` at the end
  of each call, so the BFGS objective always corresponds to the **full BCnorm
  marginal log-likelihood** on the original scale.  In `BSMclass::estim()`
  (`PTSmodel.h`) the LL is recovered with a single formula:
    `LL = -0.5 · (n_finite · log(2π) + n_finite · objFunValue)`
  Closed-form equivalent of `Σ_t bcnormLogDensityScalar(y_raw_t, μ_t,
  sqrt(σ̂²·F_t), λ)`.  Earlier versions used `n − k` for σ̂² together with the
  full-n Jacobian, which biased the Box-Cox MLE toward λ = 1 by a constant ≈
  `(n − k) · (1 − λ) · mean(log y)` whenever the state dimension was large.

- **Harmonic periods.** `lagsAll = lags / (1 : floor(lags/2))` generates all
  candidate harmonic periods.  For `lags = 12` this gives 12, 6, 4, 3, 2.4, 2.
  The C++ engine may select a subset during harmonic selection (for the `D` seasonal
  type), but `lagsAll` always stores the full candidate set passed in.
