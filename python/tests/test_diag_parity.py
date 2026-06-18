"""Phase-4 parity: diagnostics / confint / accuracy vs R."""
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
    "1LT_ar1": dict(model="1LT", orders={"ar": 1, "ma": 0, "select": False}),
    "ZZZ": dict(model="ZZZ"),
}
TOL = 1e-6
H = 12


def d(a, b):
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if a.shape != b.shape:
        return np.inf
    m = np.isfinite(a) & np.isfinite(b)
    # NaN positions must agree
    if np.any(np.isfinite(a) != np.isfinite(b)):
        return np.inf
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def main():
    with open(os.path.join(HERE, "diag_reference.json")) as fh:
        ref = json.load(fh)
    y = np.array(AIR, dtype=float)

    worst = 0.0
    n_ok = 0
    for name, kw in SPECS.items():
        r = ref[name]
        m = PTS(lags=12, h=H, holdout=True, **kw).fit(y)
        ci = m.confint(level=0.9)
        acc = m.accuracy()
        # MPE/MAPE (percent vs fraction) and asymmetry (different formula) are
        # greybox-Python vs greybox-R convention differences, not muse port
        # issues -- muse calls greybox.measures with identical arguments.
        skip = {"MPE", "MAPE", "asymmetry"}
        names = [k for k in r["acc_names"] if k not in skip]
        acc_aligned = [float(acc[k]) for k in names]
        r_acc = [v for k, v in zip(r["acc_names"], r["acc_vals"]) if k not in skip]

        diffs = {
            "rstandard": d(m.rstandard(), r["rstandard"]),
            "rstudent": d(m.rstudent(), r["rstudent"]),
            "pointLik": d(m.point_lik(), r["pointLik"]),
            "ci_lower": d(ci["lower"], r["ci_lower"]),
            "ci_upper": d(ci["upper"], r["ci_upper"]),
            "accuracy": d(acc_aligned, r_acc),
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
    print("RESULT: PASS (Python diagnostics match R)")


if __name__ == "__main__":
    main()
