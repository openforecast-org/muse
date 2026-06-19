"""Phase-5 polish parity: summary() coef table + variance proportions vs R,
including the concentrated-variance analytical SE and the G-trend Slope row."""
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
    "1DT": dict(model="1DT"),
    "1GT": dict(model="1GT"),
    "ZZZ": dict(model="ZZZ"),
    "1LT_ar1": dict(model="1LT", orders={"ar": 1, "ma": 0, "select": False}),
}
TOL = 1e-6


def _coerce(x):
    # R/jsonlite encodes NA as the string "NA" (or null); map to nan.
    return [np.nan if (v is None or v == "NA") else float(v)
            for v in np.atleast_1d(x)]


def d(a, b):
    a = np.atleast_1d(np.asarray(_coerce(a), dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(_coerce(b), dtype=float)).ravel()
    if a.shape != b.shape:
        return np.inf
    if np.any(np.isfinite(a) != np.isfinite(b)):
        return np.inf  # NaN positions must agree
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def main():
    with open(os.path.join(HERE, "summary_reference.json")) as fh:
        ref = json.load(fh)
    y = np.array(AIR, dtype=float)

    worst = 0.0
    n_ok = 0
    for name, kw in SPECS.items():
        r = ref[name]
        s = PTS(lags=12, **kw).fit(y).summary()
        c = s["coefficients"]
        p = s["proportions"]

        names_ok = list(c["names"]) == list(r["coef_names"])
        pnames_ok = list(p["names"]) == list(r["prop_names"])
        diffs = {
            "estimate": d(c["estimate"], r["estimate"]),
            "std_error": d(c["std_error"], r["std_error"]),
            "lower": d(c["lower"], r["lower"]),
            "upper": d(c["upper"], r["upper"]),
            "proportion": d(p["proportion"], r["proportion"]),
            "prop_se": d(p["std_error"], r["prop_se"]),
        }
        cworst = max(diffs.values())
        worst = max(worst, cworst)
        ok = cworst <= TOL and names_ok and pnames_ok
        n_ok += ok
        flagged = {k: v for k, v in diffs.items() if v > TOL}
        extra = ""
        if not names_ok:
            extra += f" coef_names(R={r['coef_names']} py={c['names']})"
        if not pnames_ok:
            extra += " prop_names_mismatch"
        print(f"  [{'OK' if ok else 'FAIL'}] {name:9s} worst={cworst:.2e}"
              + (f" {flagged}" if flagged else "") + extra)

    print(f"\n{n_ok}/{len(SPECS)} specs match R within {TOL:g}; worst={worst:.3e}")
    if n_ok != len(SPECS):
        sys.exit(1)
    print("RESULT: PASS (Python summary matches R)")


if __name__ == "__main__":
    main()
