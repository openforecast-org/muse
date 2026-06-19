"""Phase-0 parity test: the pybind11 engine binding must reproduce the R
engine's outputs to ~1e-6 on byte-identical inputs (dumped by
dump_reference.R into reference.json)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from muse import _musecore  # noqa: E402


def _vec(x):
    # length-1 inputs are JSON-unboxed to scalars; restore them to 1-D.
    return np.atleast_1d(np.asarray(x, dtype=float))


def run_case(inp):
    u = np.asarray(inp["u"], dtype=float).reshape(
        (inp["u_nrow"], inp["u_ncol"]), order="F"
    )
    return _musecore.ucomp(
        inp["command"],
        _vec(inp["y"]),
        u,
        inp["model"],
        int(inp["h"]),
        float(inp["lambda"]),
        float(inp["outlier"]),
        bool(inp["tTest"]),
        inp["criterion"],
        _vec(inp["periods"]),
        _vec(inp["rhos"]),
        bool(inp["verbose"]),
        bool(inp["stepwise"]),
        _vec(inp["p0"]),
        bool(inp["armaFlag"]),
        _vec(inp["TVP"]),
        float(inp["seas"]),
        inp["trendOptions"],
        inp["seasonalOptions"],
        inp["irregularOptions"],
        int(inp["nsim"]),
        int(inp["seed"]),
        float(inp["lambdaLower"]),
    )


def main():
    with open(os.path.join(HERE, "reference.json")) as fh:
        ref = json.load(fh)

    worst = 0.0
    n_ok = 0
    for name, case in ref.items():
        out = run_case(case["inputs"])
        r = case["r"]

        # model string
        assert out["model"] == r["model"], f"{name}: model {out['model']} != {r['model']}"

        # coefficients
        p_py = np.asarray(out["p"], dtype=float)
        p_r = np.asarray(r["p"], dtype=float)
        assert p_py.shape == p_r.shape, f"{name}: coef shape {p_py.shape} != {p_r.shape}"
        dcoef = float(np.max(np.abs(p_py - p_r))) if p_py.size else 0.0

        # criteria (logLik, AIC, BIC, AICc, ...)
        c_py = np.asarray(out["criteria"], dtype=float)
        c_r = np.asarray(r["criteria"], dtype=float)
        dcrit = float(np.max(np.abs(c_py - c_r))) if c_py.size else 0.0

        # lambda + objective
        dlam = abs(float(out["lambda"]) - float(r["lambda"]))
        dobj = abs(float(out["objFunValue"]) - float(r["objFunValue"]))

        cworst = max(dcoef, dcrit, dlam, dobj)
        worst = max(worst, cworst)
        status = "OK" if cworst <= 1e-6 else "FAIL"
        if cworst <= 1e-6:
            n_ok += 1
        print(
            f"  [{status}] {name:14s} model={out['model']:18s} "
            f"dcoef={dcoef:.2e} dcrit={dcrit:.2e} dlam={dlam:.2e} dobj={dobj:.2e}"
        )

    print(f"\n{n_ok}/{len(ref)} cases within 1e-6; worst abs diff = {worst:.3e}")
    if worst > 1e-6:
        sys.exit(1)
    print("RESULT: PASS (Python engine matches R)")


if __name__ == "__main__":
    main()
