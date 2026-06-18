# muse → Python port plan

Status: planning.  This document is the roadmap for translating `muse` (the PTS /
Power-Trend-Seasonal state-space package) from R to Python, modelled on the
`smooth` package's existing Python port of ADAM.

It assumes familiarity with `ARCHITECTURE.md` (the R/C++ structure) and with the
Python `smooth` package at `~/Python/Libraries/smooth/python`.

---

## 1. Guiding decision: bind the existing C++ engine, do not reimplement it

The single most important architectural fact is that **the muse C++ engine is
already front-end agnostic**.  `src/musecore.h` opens with:

> *"no Rcpp / pybind11 / R / Python dependency. The R binding lives in
> musecpp2R.cpp"*

and the only translation unit that touches `Rcpp` / `SEXP` is `src/musecpp2R.cpp`
(13 Rcpp references; zero in every engine header).  Everything below the seam —
`MuseInputs` / `MuseOutputs` / `runMuseCommand()` and the whole Kalman-filter /
BFGS / component machinery — speaks only Armadillo + STL.  Simulation uses
`arma::randn()`, not R's RNG.

Therefore the port is **two front-ends over one engine**, exactly the shape the
Python `smooth` package already uses (its `_adamCore` pybind11 module wraps an
Armadillo C++ core via `carma`):

```
            ┌──────────────────────────────┐
            │  engine (unchanged C++ core)  │
            │  musecore.h / PTSmodel.h /    │
            │  SSpace.h / boxcox.h / ...    │
            └──────────────┬───────────────┘
            MuseInputs/Outputs, runMuseCommand()
          ┌─────────────────┴─────────────────┐
   src/musecpp2R.cpp                    src/musecpp2py.cpp   ← NEW
   (Rcpp ↔ SEXP, exists)                (pybind11 ↔ numpy, to write)
          │                                     │
      R: pts()                           Python: PTS(...).fit()
```

Reimplementing the engine in NumPy is explicitly **out of scope** — it is ~5000
lines of numerically delicate state-space code (concentrated likelihood, dynamic
variance switching, anchor-snap, harmonic selection) that already exists and is
tested.  We bind it.

### File organisation mirrors smooth's three-part split

smooth separates engine / R-binding / Python-binding as:

| Role | smooth | muse (today → after port) |
|------|--------|---------------------------|
| Engine (header-only) | `src/headers/adamCore.h` | `src/musecore.h` + `PTSmodel.h` + `SSpace.h` + … (unchanged) |
| R binding | `src/adamGeneral.cpp` (RCPP_MODULE) | `src/musecpp2R.cpp` (unchanged) |
| Python binding | `src/python/adamPython.cpp` (PYBIND11_MODULE) | **`src/python/musecpp2py.cpp`** (new) |

R's build compiles only `src/*.cpp`, so a `src/python/` subfolder is invisible to
`R CMD build` and the two front-ends coexist — exactly how smooth does it.

**Binding style — keep muse's command dispatch, do not refactor to a class.**
smooth binds a *class* (`adamCore` with `.fit`/`.forecast` methods exposed via
`RCPP_MODULE` / pybind11 `class_<>`).  muse instead has a single stateless
entry point `UCompC(command, …) → runMuseCommand(MuseInputs, MuseOutputs)`.
For the port we keep muse's style: the seam already exists and is clean, and
`forecastOnly` re-runs the filter from stored params so there is no need for
persistent C++ state.  Wrapping `runMuseCommand` in a `PTSCore` class to mimic
smooth's object API is a separable, optional follow-up — not required.

### What has to change in C++ (small)

- **New file `src/python/musecpp2py.cpp`** — a pybind11 module mirroring
  `musecpp2R.cpp` field-for-field: numpy/`carma` → `MuseInputs`, call
  `runMuseCommand`, `MuseOutputs` → a Python dict.  This is the bulk of the C++
  work and it is mechanical (no class refactor; same command-string dispatch).
  Bind `UCompARMAC` here too (needed by the R-side ARMA selector port).
- **RNG**: under RcppArmadillo, `arma::randn` is remapped to R's RNG via
  `ARMA_RNG_ALT`; under pybind11 it falls back to Armadillo's default
  (`std::mt19937_64`).  Functionally fine; seeds will not match R (already an
  accepted caveat in the smooth Python port — see its `CLAUDE.md` "RNG
  Differences").  Add an explicit `arma_rng::set_seed(seed)` call in the
  simulate path so the Python `seed` argument is load-bearing.
- **No engine logic changes.**  If anything in an engine header still implicitly
  assumes R (none found in the audit), it gets lifted behind a small
  platform-neutral shim in `musecore.h`.

### Build toolchain (copy from smooth-Python)

`pyproject.toml` with `scikit-build-core` + `pybind11` + `numpy`; `CMakeLists.txt`
fetching `carma` (or vendoring `src/libs/carma` as smooth does) to convert
`arma::mat`/`arma::vec` ↔ `numpy.ndarray`.  Link LAPACK/BLAS exactly as
`src/Makevars` does today.

---

## 2. Target Python API (sklearn-style, mirroring `smooth.ADAM`)

The Python `smooth` package exposes estimators as classes with
`fit(y, X) → self` and `predict(h, ...) → ForecastResult`, plus properties
(`fitted`, `residuals`, `states`, `coef`, `scale`, `n_param`, …) and methods
(`summary`, `rstandard`, `rstudent`, `outlierdummy`, `simulate`).  muse should
mirror this so the two libraries feel identical.

```python
from muse import PTS

m = PTS(model="ZZZ", lags=12, ic="BICc", h=12, holdout=True).fit(y)
m.summary()
fc = m.predict(h=12, interval="prediction", level=[0.8, 0.95])
fc.plot()
```

### Constructor parameters (map from `pts()` formals)

| R `pts()` arg | Python `PTS(...)` | Notes |
|---------------|-------------------|-------|
| `data` (in `pts()`) | `y` in `.fit(y, X=None)` | sklearn split: spec in ctor, data in fit |
| `model="ZZZ"` | `model="ZZZ"` | same 3-letter PTS grammar |
| `lags` | `lags=None` | infer from pandas index when `None` (per smooth rule: **never** add a `frequency` param) |
| `orders` | `orders=None` (dict) or `ar/ma/i/select` kwargs | mirror ADAM's `ar_order/ma_order/arima_select` style |
| `formula` | dropped | use `X` ndarray / DataFrame columns instead (ADAM has no formula) |
| `regressors="use"` | `regressors="use"` | only `"use"` implemented, as in R |
| `outliers="ignore"` | `outliers="ignore"` | `"use"` supported; `"select"` errors (same as R) |
| `level=0.99` | `outliers_level=0.99` | name aligned with ADAM |
| `ic="BICc"` | `ic="BICc"` | **default BICc** (muse-specific; ADAM defaults AICc) |
| `h`, `holdout` | `h=None`, `holdout=False` | same |
| `verbose` | `verbose=0` | int level, ADAM convention |
| `B=...` (in `...`) | `initial=` / `b0=` | optimiser warm-start |

### Methods / properties (map from the S3 method inventory)

| R S3 method | Python | Kind | Engine call? |
|-------------|--------|------|--------------|
| `pts()` | `PTS().fit()` | method | `command="all"` |
| `forecast.pts` | `.predict()` | method | `command="forecastOnly"` |
| `predict.pts` (= fitted) | `.fitted` | property | — |
| `simulate.pts` | `.simulate()` | method | `command="simulate"` |
| `fitted` / `residuals` / `actuals` | `.fitted` / `.residuals` / `.actuals` | property | — |
| `coef` / `vcov` | `.coef` / `.vcov` | property | — |
| `nobs` / `nparam` / `sigma` / `logLik` | `.nobs` / `.n_param` / `.sigma` / `.log_lik` | property | — |
| `confint` | `.confint()` | method | — |
| `summary` / `print` | `.summary()` / `__repr__` | method | — |
| `rstandard` / `rstudent` / `pointLik` | `.rstandard()` / `.rstudent()` / `.point_lik()` | method | — |
| `outlierdummy` | `.outlierdummy()` | method | — |
| `accuracy` | `.accuracy()` | method | greybox `measures` |
| `orders` / `lags` / `modelType` / `errorType` | properties | property | — |
| `initials` / `pls` | `.initials` / `.pls()` | property/method | `pls` may re-fit |
| `update` | `.update()` | method | re-fit |
| `plot.pts` | `.plot()` + `fc.plot()` | method | matplotlib (port `plotting.py`) |
| `AIC`/`BIC`/`AICc`/`BICc` (dispatched) | `.aic`/`.bic`/`.aicc`/`.bicc` | property | local (no smooth fallthrough in Python) |

Note: R gets `plot`, `AIC`, `print.adam` etc. *for free* via the `c("pts","smooth")`
class chain.  Python has no S3 dispatch chain, so each becomes an explicit
method/property on `PTS` — small, but must be written, not inherited.

### `ForecastResult`

Reuse / mirror smooth-Python's `forecaster/result.py` `ForecastResult`
(`.mean`, `.lower`, `.upper`, `.level`, `.interval`, `.scenarios`, `.plot()`).
muse-specific: bounds are endpoint-transformed through the inverse Box-Cox (BC
support boundary handling), not symmetric.

---

## 3. What ports as pure Python (no engine involvement)

Everything in the R post-processing / orchestration layer is pure Python +
NumPy/SciPy/pandas.  Dependencies already exist:

- **`smooth.msdecompose`** — already implemented in Python
  (`core/utils/utils.py`).  Directly reusable by the λ screen.
- **greybox-Python** — provides `measures` (`point_measures.py`) for
  `accuracy`, `alm`, and the `bcnorm` family for forecast quantiles.

| R file | Python target | Difficulty | Depends on |
|--------|---------------|------------|------------|
| `pts.R` (orchestration) | `core/pts/pts.py` (the `PTS` class `fit`) | M | engine, translate, screen |
| `.pts_guerrero_decomp_lambda` (`pts-internals.R`) | `core/pts/lambda_screen.py` | **S** | `smooth.msdecompose`, `scipy.optimize.minimize_scalar` |
| `pts-translate.R` (`pts_to_uc`, `uc_to_pts`, `uc_to_arma`) | `core/pts/translate.py` | S | string ops only |
| `.pts_select_pts_arma` / `.pts_select_arma_at_lag` | `core/pts/selector.py` | M | engine (`command="all"`, `UCompARMAC`) |
| `.pts_parse_data` / wrappers (`pts-internals.R`) | `core/pts/io.py` | M | pandas index ↔ ts/zoo semantics |
| `.inv_box_cox` | `core/pts/boxcox.py` (or reuse greybox) | S | — |
| `methods.R` forecast interval logic | `core/pts/forecaster.py` | M | engine, bcnorm quantiles |
| `pts-summary.R` (coef table, var proportions, delta-method SE) | `.summary()` | M | NumPy linear algebra |
| `pts-accuracy.R` | `.accuracy()` | S | greybox `measures` |
| `pts-simulate.R` | `.simulate()` | M | engine (`command="simulate"`) |
| `pts-confint.R` | `.confint()` | S | — |
| `pts-methods-diagnostics.R` | diagnostics methods | S | greybox bcnorm / scipy |

`UCompARMAC` (the residual-ARMA scorer used by the selector) is a second exported
C++ entry alongside `UCompC`; it must be bound in `musecpp2py.cpp` too.

---

## 4. Portability assessment — can everything be implemented?

**Yes, with three things to handle deliberately:**

1. **Time index semantics (ts/zoo → pandas).**  R carries `ts`/`zoo` time
   indices through every wrapper (`.pts_wrap_in/oos/states`).  Python uses a
   pandas `DatetimeIndex` / `RangeIndex`.  smooth-Python already solved this
   pattern; reuse its index-inference utilities.  This is the single most
   pervasive (if individually small) translation chore.

2. **Concentrated-variance `vcov` NaN semantics.**  The engine returns `NaN`
   rows/cols for concentrated / constrained parameters; `summary` marks them
   with `*` and `confint` warns.  Must be preserved exactly so ICs and SEs match
   R.  Pure carry-through of the engine output — no recomputation.

3. **RNG non-reproducibility R↔Python for `simulate` / simulated intervals.**
   Accepted and documented (same as smooth-Python).  Validation of simulation
   paths is statistical (distributional), not bit-exact; everything else
   (estimation, forecasts, ICs) is deterministic and **must** match R to
   ~1e-6 on shared CSV inputs.

Nothing in the muse method surface requires a capability Python lacks.  The
`formula` interface is the only feature intentionally dropped (ADAM has no
formula API; use `X`).

### Tension to resolve: the `[0, 2]` λ clip vs smooth-Python's "never clip" rule

smooth-Python's `CLAUDE.md` forbids clipping/clamping model numerics.  muse's λ
screen deliberately *bounds* the Guerrero optimisation to `[0, 2]`.  These are not
in conflict: the clip is a **domain restriction on a hyperparameter search**
(a documented modelling choice that removes the inverse-BC asymptote), not an
in-place silencing of a model output or a `pmax` inside a `log`.  It stays.  The
plan flags it so a reviewer porting under the smooth rules does not "fix" it away.

---

## 5. Package skeleton (mirror smooth-Python)

```
muse/python/
├── pyproject.toml                 # scikit-build-core + pybind11 + numpy
├── CMakeLists.txt                 # fetch/vendor carma, link LAPACK/BLAS
├── NEWS.md                        # required changelog (smooth convention)
├── CLAUDE.md                      # py-specific build/test guidance
├── src/
│   └── muse/
│       ├── __init__.py            # exports PTS
│       ├── _musecore.*.so         # built pybind11 module (UCompC, UCompARMAC)
│       └── core/
│           └── pts/
│               ├── pts.py         # PTS class (fit/predict/properties/summary)
│               ├── translate.py   # pts_to_uc / uc_to_pts / uc_to_arma
│               ├── lambda_screen.py
│               ├── selector.py    # PTS-then-ARMA selection
│               ├── forecaster.py  # interval logic + ForecastResult
│               ├── boxcox.py
│               ├── io.py          # pandas index handling
│               └── plotting.py
└── tests/
    └── ...                        # R↔Python parity tests on shared CSVs
```

C++ binding source lives in `src/python/musecpp2py.cpp` (mirroring smooth's
`src/python/adamPython.cpp`), so the two front-ends share the engine headers in
`src/` verbatim while R's build ignores the `src/python/` subfolder.

---

## 6. Phased roadmap

**Phase 0 — Engine binding spike (highest risk first).**
Write a minimal `src/musecpp2py.cpp` exposing `UCompC("all", ...)` only.  Build
with scikit-build-core + carma.  Goal: fit one fixed-spec model (`"1NN"` on
AirPassengers) from Python and match R's `coef`/`logLik` to 1e-6.  This de-risks
the entire project (toolchain, carma marshalling, LAPACK linking, RNG fallback).

**Phase 1 — Estimation path.**
`PTS` class + `fit()` for fixed specs; `translate.py`; `io.py` index handling;
core properties (`fitted`, `residuals`, `coef`, `vcov`, `scale`, `n_param`,
`log_lik`, ICs).  Parity tests vs R on a CSV battery.

**Phase 2 — λ screen + selection.**
Port `lambda_screen.py` (reuse `smooth.msdecompose`) and `selector.py`
(PTS-then-ARMA).  Wire `model="ZZZ"` and `orders=...select=True`.  Bind
`UCompARMAC`.  Verify λ and selected structure match R.

**Phase 3 — Forecasting + intervals.**
`forecaster.py` (`command="forecastOnly"`), `ForecastResult`, all interval types
(prediction / confidence / simulated / none), `side`, `cumulative`, vector
`level`, inverse-BC endpoint transform.  Bind greybox-Python bcnorm quantiles.

**Phase 4 — Diagnostics, summary, accuracy, simulate, update.**
`summary()` (coef table + variance proportions + delta-method SEs), `rstandard`,
`rstudent`, `point_lik`, `outlierdummy`, `confint`, `accuracy` (greybox
`measures`), `simulate`, `update`.

**Phase 5 — Outliers, plotting, polish.**
`outliers="use"` path (incl. the λ-pinning workaround), `plotting.py`, docs,
`NEWS.md`, packaging.

---

## 7. Validation strategy

Per smooth-Python's documented convention (R and Python RNGs differ): generate
inputs in R, save to CSV, load the *same* CSV in both languages, and compare.

- **Deterministic parity (must match ~1e-6):** `coef`, `vcov`, `logLik`, all
  ICs, `fitted`, `residuals`, `forecast$mean`, prediction-interval bounds, the
  screened λ, the selected structure.
- **Statistical parity only:** `simulate` paths and simulated intervals
  (distributional checks, not bit-exact).
- Reuse the existing R `tests/testthat` series as the CSV battery so the two
  test suites cover the same cases.

---

## 8. Dependencies summary

| Need | Source | Status |
|------|--------|--------|
| Armadillo ↔ numpy | `carma` | vendored in smooth-Python; reuse |
| Build | `scikit-build-core`, `pybind11`, CMake | pattern established in smooth-Python |
| `msdecompose` | `smooth` (Python) | already ported |
| `measures` (accuracy) | `greybox` (Python) | exists (`point_measures.py`) |
| bcnorm density/quantiles | `greybox` (Python) | exists (`distributions/`) — verify `qbcnorm` λ<0 truncation parity with R fix |
| numerics | NumPy / SciPy / pandas | standard |

The only dependency requiring a parity audit is greybox-Python's `bcnorm`
quantile (`qbcnorm`) for λ < 0 — the R side was recently fixed to renormalise
against the truncated distribution; confirm the Python port carries the same fix
before relying on it for forecast intervals.
