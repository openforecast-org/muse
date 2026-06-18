"""Input parsing + pandas index handling for PTS.

Accepts a plain array or a pandas Series; when the Series carries a
DatetimeIndex the seasonal period is inferred from its frequency (mirroring
smooth's convention -- never expose a `frequency` parameter), and the index
is carried through to fitted / residuals / forecast outputs.
"""
from __future__ import annotations

import numpy as np

# freq prefix -> seasonal period (mirrors smooth.adam_general ... sma.py)
_FREQ_MAP = {
    "MS": 12, "ME": 12, "M": 12,
    "QS": 4, "QE": 4, "Q": 4,
    "W": 52, "D": 7, "h": 24, "H": 24,
}


def parse_input(y):
    """Return (values, index, inferred_lags).  index is None for plain arrays."""
    try:
        import pandas as pd
        if isinstance(y, pd.Series):
            values = y.to_numpy(dtype=float).ravel()
            index = y.index
            inferred = None
            if isinstance(index, pd.DatetimeIndex):
                freq = index.freq or pd.infer_freq(index)
                if freq is not None:
                    fs = str(getattr(freq, "freqstr", freq))
                    for k in sorted(_FREQ_MAP, key=len, reverse=True):
                        if fs.startswith(k) or fs.startswith("<" + k):
                            inferred = _FREQ_MAP[k]
                            break
            return values, index, inferred
    except ImportError:
        pass
    return np.asarray(y, dtype=float).ravel(), None, None


def future_index(index, h):
    """Extend a pandas/array index by h steps for forecast output."""
    if index is None:
        return None
    try:
        import pandas as pd
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freq or pd.infer_freq(index)
            return pd.date_range(index[-1], periods=h + 1, freq=freq)[1:]
        if isinstance(index, pd.RangeIndex):
            step = index.step
            start = index[-1] + step
            return pd.RangeIndex(start, start + step * h, step)
    except ImportError:
        return None
    last = index[-1]
    step = (index[1] - index[0]) if len(index) > 1 else 1
    return np.array([last + step * (i + 1) for i in range(h)])


def wrap(values, index):
    """Wrap a 1-D result as a pandas Series if an index is available."""
    values = np.asarray(values)
    if index is None:
        return values
    try:
        import pandas as pd
        return pd.Series(values, index=index[: len(values)])
    except ImportError:
        return values
