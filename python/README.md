# muse

[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Python CI](https://github.com/config-i1/muse/actions/workflows/python-check.yaml/badge.svg)](https://github.com/config-i1/muse/actions/workflows/python-check.yaml)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

<img src="https://github.com/config-i1/muse/raw/main/man/figures/muse-purple-light-web.png" alt="muse logo" width="220" align="right" />

Python implementation of the **muse** package (**M**ultiple **U**nobserved **S**ources of **E**rror) for the MSOE state-space models for time series analysis and forecasting.

Currently, the package only implements the PTS model, which is a multiple-source-of-error (MSOE) structural state-space model: a Kalman filter / smoother runs over a level / trend / seasonal / irregular decomposition, with an optional Box-Cox power transform, and the components are selected analogously to the ETS taxonomy, based on information criteria. The estimation engine is written in C++ (Armadillo), binding through `pybind11`.

## Features

- **Single entry point** — the `PTS` class. Spec in the constructor, data into `.fit()`, results via properties and `.predict()` (scikit-learn style).
- **Compact model strings** — `"PTS"` = *Power / Trend / Seasonal*, e.g. `"ZZZ"`, `"0LT"`, `"0.5GD"` (see [Model specification](#model-specification)).
- **Automatic model selection** — `"Z"` in any position auto-selects the Box-Cox power (variance-stabilisation screen), the trend, the seasonal type, and (optionally) the ARMA orders of the irregular component, by information criterion (AICc / AIC / BIC / **BICc**, the default).
- **Box-Cox transform** — fixed (`"0"`, `"0.5"`, `"1"`, …) or estimated (`"Z"`), with proper back-transformation of fitted values and forecasts.
- **ARMA / SARMA irregular** — non-seasonal `orders={"ar": p, "ma": q}`, or seasonal `orders={"ar": [p, P], "ma": [q, Q]}` (the seasonal lag comes from the top-level `lags`, not from `orders`), or an automatic `select=True` order search.
- **Forecasts with intervals** — prediction, confidence, simulated, or none; one- or two-sided; vector confidence levels; cumulative forecasts.
- **Missing values treatment** — done automatically via the filtering and smoothing, embedded in the model.
- **Outlier detection** — `outliers="use"` detects AO / LS / SC events during estimation and reports them."use"` detects AO / LS / SC events during estimation and reports them.
- **Diagnostics & accuracy** — `summary()`, `rstandard()`, `rstudent()`,
  `point_lik()`, `confint()`, `accuracy()`, `simulate()`, `update()`.
- **pandas-aware** — pass a `pandas.Series` with a `DatetimeIndex` and the
  seasonal period is inferred; fitted values / residuals / forecasts come back
  indexed.
- **Diagnostic plots** — `plot()` reuses the `smooth` plotting suite.

## Installation

> Not yet on PyPI. Install from source (the C++ extension is built on install).

```bash
pip install "git+https://github.com/config-i1/muse.git@python#subdirectory=python"
```

Or from a local checkout:

```bash
git clone https://github.com/config-i1/muse.git
pip install ./muse/python
```

## System requirements

Installing from source compiles a C++ extension. You will need:

- a **C++17 compiler** (g++, clang++, or MSVC),
- **CMake** ≥ 3.16,
- the **Armadillo** headers and a **LAPACK / BLAS** library.

On Debian / Ubuntu:

```bash
sudo apt-get install libarmadillo-dev liblapack-dev libblas-dev
```

### Optional dependencies

| Feature | Needs | Notes |
|---|---|---|
| Auto Box-Cox `λ` (power `"Z"`) | [`smooth`](https://github.com/config-i1/smooth) | uses `msdecompose` for the variance-stabilisation screen |
| `accuracy()` error measures | [`greybox`](https://github.com/config-i1/greybox) | uses `measures` |
| `plot()` | `smooth`, `matplotlib` | reuses the `smooth` diagnostic plots |

`numpy`, `scipy`, and `pandas` are installed automatically.

## Quick example

```python
import numpy as np
import pandas as pd
from muse import PTS

# monthly series with a DatetimeIndex -> seasonal period inferred (12)
y = pd.Series(my_values, index=pd.date_range("2010-01-01", periods=120, freq="MS"))

# fully automatic: Box-Cox power, trend, seasonal type all selected by BICc
model = PTS(model="ZZZ", h=12, holdout=True).fit(y)

model.summary()                 # coefficient table + variance proportions
print(model.model_label)        # e.g. "PTS(0,L,T)"
print(model.aicc, model.bicc)   # information criteria

# forecast 12 steps with 80% and 95% prediction intervals
fc = model.predict(12, interval="prediction", level=[0.8, 0.95])
print(fc.mean, fc.lower, fc.upper)

model.plot()                    # diagnostic plots
```

A fixed specification with an ARMA(1,0) irregular and no transform:

```python
m = PTS(model="1LT", lags=12, orders={"ar": 1, "ma": 0}).fit(y)
m.predict(12)
```

A seasonal SARMA(1,0)(1,0)_12 irregular — `ar`/`ma` are length-2 vectors and
the seasonal lag (12) is read from the top-level `lags`:

```python
m = PTS(model="1LT", lags=12, orders={"ar": [1, 1], "ma": [0, 0]}).fit(y)
```

## Model specification

The three-character `model` string encodes **P**ower / **T**rend / **S**easonal:

| Position | Letter | Meaning |
|---|---|---|
| **Power** (Box-Cox λ) | a number (`"0"`, `"0.5"`, `"1"`, …) | fixed λ (`0` = log, `1` = no transform) |
| | `Z` | estimate λ from the data |
| **Trend** | `N` | none (random walk / level only) |
| | `L` | local linear (level + slope) |
| | `D` | damped trend |
| | `G` | global / deterministic trend |
| | `Z` | auto-select |
| **Seasonal** | `N` | none |
| | `D` | discrete (linear / harmonic) |
| | `T` | trigonometric (equal-variance harmonics) |
| | `Z` | auto-select |

Examples: `"ZZZ"` (everything automatic), `"0LT"` (log, local-linear, trigonometric),
`"1NN"` (no transform, random walk), `"0.5GD"` (square-root, deterministic trend,
discrete seasonal).

## API at a glance

```python
PTS(model="ZZZ", lags=None, orders=None, ic="BICc",
    outliers="ignore", level=0.99, h=0, holdout=False, verbose=False)
```

| Method | Purpose |
|---|---|
| `.fit(y, X=None)` | estimate the model (`y` = array or `pandas.Series`) |
| `.predict(h, interval=, level=, side=, cumulative=, ...)` | forecast → `ForecastResult` |
| `.summary()` | coefficient table + variance proportions |
| `.plot(which=(1,2,4,6))` | diagnostic plots |
| `.accuracy(holdout=None)` | holdout error measures |
| `.simulate(nsim, seed)` | in-sample replay paths |
| `.update(**overrides)` | re-fit with changed settings |
| `.rstandard()` / `.rstudent()` / `.point_lik()` / `.confint()` / `.outlierdummy()` | diagnostics |

| Property | |
|---|---|
| `fitted`, `residuals`, `actuals` | series (indexed if `y` was a `Series`) |
| `coef`, `coef_values`, `coef_names`, `vcov` | parameters |
| `lambda_`, `model_label`, `orders`, `outliers_detected` | resolved spec |
| `nobs`, `n_param`, `log_lik`, `scale`, `sigma` | fit summary |
| `aic`, `bic`, `aicc`, `bicc` | information criteria |

## Relationship to R

The C++ engine is the same code that powers the R `muse` package; the Python
front-end is a thin `pybind11` binding plus an API layer. Every deterministic
output (coefficients, log-likelihood, information criteria, fitted values,
forecasts and intervals, diagnostics, the summary table) matches the R
`muse::pts()` output to within `1e-6` — most to machine precision.

## See also

- [muse on GitHub](https://github.com/config-i1/muse) — R + Python sources
- [smooth](https://github.com/config-i1/smooth) — the ADAM / ETS / ARIMA family
  (the Python API muse mirrors)
- [greybox](https://github.com/config-i1/greybox) — distributions, information
  criteria, and error measures

## Acknowledgements

The Python translation of the package — and parts of the underlying C++ engine
refactoring — were developed with the assistance of Anthropic's Claude.
Responsibility for the code and its correctness rests with the package authors.

## License

LGPL-2.1. See [LICENSE](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html).
