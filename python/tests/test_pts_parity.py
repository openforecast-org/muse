"""Phase-1 end-to-end parity: the Python PTS class vs R's pts() on
byte-identical data (AirPassengers), for fixed-lambda specs."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from muse import PTS  # noqa: E402

# AirPassengers, hard-coded so Python and R use identical input.
AIR = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]

SPECS = {
    "1NN": dict(model="1NN"),
    "1LN": dict(model="1LN"),
    "1LT": dict(model="1LT"),
    "1DT": dict(model="1DT"),
    "1ND": dict(model="1ND"),
    "0LT": dict(model="0LT"),
    "0.5LT": dict(model="0.5LT"),
    "1GT": dict(model="1GT"),
    "1ZZ": dict(model="1ZZ"),
    "1ZN": dict(model="1ZN"),
    "1LT_ar1": dict(model="1LT", orders={"ar": 1, "ma": 0, "select": False}),
    "1LT_ma1": dict(model="1LT", orders={"ar": 0, "ma": 1, "select": False}),
    "ZNN": dict(model="ZNN"),
    "ZLT": dict(model="ZLT"),
    "ZZZ": dict(model="ZZZ"),
}

TOL = 1e-6


def main():
    with open(os.path.join(HERE, "pts_reference.json")) as fh:
        ref = json.load(fh)

    y = np.array(AIR, dtype=float)
    worst = 0.0
    n_ok = 0
    for name, kw in SPECS.items():
        r = ref[name]
        m = PTS(lags=12, **kw).fit(y)

        diffs = {}
        diffs["lambda"] = abs(m.lambda_ - float(r["lambda"]))
        cp = m.coef_values
        cr = np.atleast_1d(np.asarray(r["coef"], dtype=float))
        diffs["coef"] = (
            float(np.max(np.abs(cp - cr))) if cp.shape == cr.shape else np.inf
        )
        for key, val in [
            ("logLik", m.log_lik), ("AIC", m.aic), ("BIC", m.bic),
            ("AICc", m.aicc), ("BICc", m.bicc), ("sigma", m.sigma),
        ]:
            diffs[key] = abs(val - float(r[key]))
        diffs["nParam"] = abs(m.n_param - int(r["nParam"]))
        diffs["nobs"] = abs(m.nobs - int(r["nobs"]))
        fp = np.asarray(m.fitted, dtype=float)
        fr = np.atleast_1d(np.asarray(r["fitted"], dtype=float))
        diffs["fitted"] = (
            float(np.max(np.abs(fp - fr))) if fp.shape == fr.shape else np.inf
        )
        rp = np.asarray(m.residuals, dtype=float)
        rr = np.atleast_1d(np.asarray(r["residuals"], dtype=float))
        diffs["residuals"] = (
            float(np.max(np.abs(rp - rr))) if rp.shape == rr.shape else np.inf
        )

        model_ok = m.model_label == r["model"]
        cworst = max(diffs.values())
        worst = max(worst, cworst)
        ok = cworst <= TOL and model_ok
        n_ok += ok
        flagged = {k: v for k, v in diffs.items() if v > TOL}
        print(
            f"  [{'OK' if ok else 'FAIL'}] {name:9s} {m.model_label:14s} "
            f"worst={cworst:.2e}"
            + (f"  model_mismatch(R={r['model']})" if not model_ok else "")
            + (f"  {flagged}" if flagged else "")
        )

    print(f"\n{n_ok}/{len(SPECS)} specs match R within {TOL:g}; "
          f"worst abs diff = {worst:.3e}")
    if n_ok != len(SPECS):
        sys.exit(1)
    print("RESULT: PASS (Python PTS matches R pts())")


if __name__ == "__main__":
    main()
