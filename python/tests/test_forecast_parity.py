"""Phase-3 parity: the Python forecaster vs R forecast.pts -- mean and
prediction/confidence/one-sided intervals, fixed- and auto-lambda."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from muse import PTS  # noqa: E402
from test_pts_parity import AIR  # noqa: E402

SPECS = {
    "1LT": dict(model="1LT"),
    "0LT": dict(model="0LT"),
    "0.5LT": dict(model="0.5LT"),
    "1DT": dict(model="1DT"),
    "ZZZ": dict(model="ZZZ"),
    "1LT_ar1": dict(model="1LT", orders={"ar": 1, "ma": 0, "select": False}),
}
TOL = 1e-6
H = 12


def main():
    with open(os.path.join(HERE, "forecast_reference.json")) as fh:
        ref = json.load(fh)
    y = np.array(AIR, dtype=float)

    worst = 0.0
    n_ok = 0
    for name, kw in SPECS.items():
        r = ref[name]
        m = PTS(lags=12, **kw).fit(y)

        fp = m.predict(H, interval="prediction", level=[0.8, 0.95])
        fc = m.predict(H, interval="confidence", level=0.95)
        fu = m.predict(H, interval="prediction", level=0.95, side="upper")

        def d(a, b):
            a = np.asarray(a, dtype=float).ravel(order="F")
            b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
            if a.shape != b.shape:
                return np.inf
            m_ = np.isfinite(a) & np.isfinite(b)
            return float(np.max(np.abs(a[m_] - b[m_]))) if m_.any() else 0.0

        diffs = {
            "pred_mean": d(fp.mean, r["pred_mean"]),
            "pred_lower": d(fp.lower, r["pred_lower"]),
            "pred_upper": d(fp.upper, r["pred_upper"]),
            "conf_lower": d(fc.lower, r["conf_lower"]),
            "conf_upper": d(fc.upper, r["conf_upper"]),
            "up_lower": d(fu.lower, r["up_lower"]),
            "up_upper": d(fu.upper, r["up_upper"]),
        }
        cworst = max(diffs.values())
        worst = max(worst, cworst)
        ok = cworst <= TOL
        n_ok += ok
        flagged = {k: v for k, v in diffs.items() if v > TOL}
        print(f"  [{'OK' if ok else 'FAIL'}] {name:9s} worst={cworst:.2e}"
              + (f"  {flagged}" if flagged else ""))

    print(f"\n{n_ok}/{len(SPECS)} specs match R within {TOL:g}; "
          f"worst abs diff = {worst:.3e}")
    if n_ok != len(SPECS):
        sys.exit(1)
    print("RESULT: PASS (Python forecaster matches R forecast.pts)")


if __name__ == "__main__":
    main()
