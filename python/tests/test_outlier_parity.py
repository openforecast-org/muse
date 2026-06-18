"""Phase-5 parity: outliers='use' vs R (engine outlier detection)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from muse import PTS  # noqa: E402
from test_pts_parity import AIR  # noqa: E402

CASES = {
    "1LT_spike60": dict(model="1LT", idx=60, mult=2.0, level=0.99),
    "ZZZ_spike100": dict(model="ZZZ", idx=100, mult=1.8, level=0.99),
    "1LT_lvl95": dict(model="1LT", idx=60, mult=2.0, level=0.95),
}
TOL = 1e-6


def main():
    with open(os.path.join(HERE, "outlier_reference.json")) as fh:
        ref = json.load(fh)

    n_ok = 0
    worst = 0.0
    for name, c in CASES.items():
        r = ref[name]
        y = np.array(AIR, dtype=float)
        y[c["idx"] - 1] *= c["mult"]
        m = PTS(model=c["model"], lags=12, h=0,
                outliers="use", level=c["level"]).fit(y)

        ok = True
        ok &= m.model_label == r["model"]
        ok &= m.n_param == int(r["nParam"])
        ok &= list(m.coef_names) == list(r["coefNames"])
        det_time = [d["time"] for d in m.outliers_detected]
        det_type = [d["type"] for d in m.outliers_detected]
        ok &= det_time == list(np.atleast_1d(r["det_time"]).astype(int)) \
            if r["det_time"] is not None else det_time == []
        ok &= det_type == list(np.atleast_1d(r["det_type"])) \
            if r["det_type"] is not None else det_type == []

        dcoef = float(np.max(np.abs(
            m.coef_values - np.atleast_1d(np.asarray(r["coef"], dtype=float))
        ))) if m.coef_values.size == len(np.atleast_1d(r["coef"])) else np.inf
        dll = abs(m.log_lik - float(r["logLik"]))
        cworst = max(dcoef, dll)
        worst = max(worst, cworst)
        ok &= cworst <= TOL
        n_ok += ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name:13s} {m.model_label:14s} "
              f"nP={m.n_param} det={det_time}/{det_type} "
              f"dcoef={dcoef:.2e} dll={dll:.2e}")

    print(f"\n{n_ok}/{len(CASES)} cases match R within {TOL:g}; worst={worst:.3e}")
    if n_ok != len(CASES):
        sys.exit(1)
    print("RESULT: PASS (Python outliers='use' matches R)")


if __name__ == "__main__":
    main()
