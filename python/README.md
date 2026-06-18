# muse (Python)

Python front-end for the **muse** PTS (Power / Trend / Seasonal) state-space
forecasting engine.  The C++ engine is shared verbatim with the R package via
a pybind11 binding; the Python API mirrors `smooth.ADAM`.

```python
from muse import PTS
m = PTS(model="ZZZ", lags=12, h=12, holdout=True).fit(y)
m.summary()
fc = m.predict(12, interval="prediction", level=[0.8, 0.95])
```

In development. Requires `smooth` (msdecompose) for the auto-lambda screen and
`greybox` (measures) for `accuracy()`.
