# muse - Multiple Unobserved Sources of Error

<img src="../man/figures/muse-purple-light-web.png" alt="muse logo" width="300" />

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1)

**muse** is a Python package implementing the **PTS** (*Power / Trend / Seasonal*) state-space family of models for time-series analysis and for forecasting. The estimation engine is written in C++ (Rcpp & RcppArmadillo), wrapping a Kalman filter/smoother around a Multiple Source of Error (MSOE) model whose components are selected analogously to the ETS taxonomy.

Python front-end for the **muse** PTS (Power / Trend / Seasonal) state-space
forecasting engine.  The C++ engine is shared verbatim with the R package via
a pybind11 binding; the Python API mirrors `smooth.ADAM`.

```python
from muse import PTS
m = PTS(model="ZZZ", lags=12, h=12, holdout=True).fit(y)
m.summary()
fc = m.predict(12, interval="prediction", level=[0.8, 0.95])
```
