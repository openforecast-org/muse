"""PTS <-> UC model-string translation. Port of R/pts-translate.R.

Not part of the public API; used by PTS.fit() to turn a 3-letter PTS spec
into the UC string the C++ engine speaks, and back.
"""
from __future__ import annotations


def pts_to_uc(model: str, arma_orders=(0, 0)) -> tuple[str, float]:
    """Turn a PTS spec ("0NT", "ZZZ", "0.5LD") into (uc_string, lambda).

    Mirrors R's pts_to_uc().  arma_orders is (p, q) for a non-seasonal
    ARMA or (p, q, P, Q, s) for SARMA.
    """
    n = len(model)
    # ARMA / SARMA irregular component
    if len(arma_orders) == 2:
        arma_tok = f"/arma({int(arma_orders[0])},{int(arma_orders[1])})"
    elif len(arma_orders) == 5:
        a = [int(x) for x in arma_orders]
        arma_tok = f"/arma({a[0]},{a[1]},{a[2]},{a[3]},{a[4]})"
    else:
        raise ValueError("arma_orders must have length 2 (arma) or 5 (sarma).")

    # Seasonal (last char)
    aux = model[n - 1].lower()
    seasonal_tok = {"z": "/?", "n": "/none", "d": "/linear", "t": "/equal"}.get(aux)
    if seasonal_tok is None:
        raise ValueError(f"Invalid seasonal letter in PTS spec: '{aux}'")
    modelu = seasonal_tok + arma_tok

    # Trend (second-to-last char)
    aux = model[n - 2].lower()
    trend_tok = {
        "z": "?/none",
        "n": "rw/none",
        "l": "llt/none",
        "d": "srw/none",
        "g": "td/none",
    }.get(aux)
    if trend_tok is None:
        raise ValueError(f"Invalid trend letter in PTS spec: '{aux}'")
    modelu = trend_tok + modelu

    # Power (Box-Cox lambda): numeric or 'Z'
    aux = model[: n - 2].lower()
    try:
        lam = float(aux)
    except ValueError:
        if aux == "z":
            lam = 9999.9  # C++ sentinel for "estimate"
        else:
            raise ValueError(
                f"Invalid power letter in PTS spec: '{aux}'"
            ) from None
    return modelu, lam


def uc_to_pts(model_uc: str, lam: float) -> str:
    """Format a resolved UC string + lambda as PTS(<lambda>,<trend>,<seas>)."""
    model_uc = model_uc.replace("/none/", "/", 1)
    sl = [i for i, c in enumerate(model_uc) if c == "/"]
    trend = model_uc[: sl[0]]
    seasonal = model_uc[sl[0] + 1 : sl[1]]
    trend_letter = {"rw": "N", "srw": "D", "llt": "L", "td": "G", "?": "Z"}.get(
        trend, ""
    )
    seasonal_letter = {"none": "N", "equal": "T", "linear": "D", "?": "Z"}.get(
        seasonal, ""
    )
    return f"PTS({_fmt_lambda(lam)},{trend_letter},{seasonal_letter})"


def _fmt_lambda(lam: float) -> str:
    r = round(float(lam), 2)
    if r == int(r):
        return str(int(r))
    return str(r)


def uc_to_arma(model: str) -> dict:
    """Pull ARMA orders out of an arma(...) block in a UC string."""
    i = model.find("arma(")
    if i < 0:
        return {"ar": 0, "ma": 0, "lags": [1]}
    j = model.find(")", i)
    body = model[i + 5 : j]
    parts = []
    for tok in body.split(","):
        try:
            parts.append(int(tok))
        except ValueError:
            parts.append(0)
    if len(parts) <= 2:
        ar = parts[0] if parts else 0
        ma = parts[1] if len(parts) >= 2 else 0
        return {"ar": ar, "ma": ma, "lags": [1]}
    if len(parts) == 5:
        return {
            "ar": [parts[0], parts[2]],
            "ma": [parts[1], parts[3]],
            "lags": [1, parts[4]],
        }
    return {"ar": 0, "ma": 0, "lags": [1]}


def orders_to_uc(orders, seasonal_lag):
    """Normalise the ARMA `orders` spec into per-lag (ar, ma, lags) vectors.

    Port of R's .pts_orders_to_uc, with one deliberate difference: the
    seasonal ARMA lag comes from the top-level `lags` (seasonal_lag), NOT from
    `orders` -- `orders` carries only `ar`, `ma`, and `select`.

    `ar` / `ma` may be scalars (non-seasonal ARMA(p, q)) or length-2 vectors
    (SARMA(p, q)(P, Q)_s, where s = seasonal_lag).  Returns a dict with
    integer-list `ar`, `ma`, `lags`, and a bool `select`.
    """
    orders = orders or {}
    if "lags" in orders:
        raise ValueError(
            "`lags` does not belong in `orders`; it is the separate top-level "
            "`lags` argument (the seasonal period)."
        )
    select = bool(orders.get("select", False))

    def _ints(x):
        if x is None:
            return [0]
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x] or [0]
        return [int(x)]

    ar, ma = _ints(orders.get("ar", 0)), _ints(orders.get("ma", 0))
    L = max(len(ar), len(ma), 1)
    ar += [0] * (L - len(ar))
    ma += [0] * (L - len(ma))
    if any(v < 0 for v in ar) or any(v < 0 for v in ma):
        raise ValueError("`orders` ar/ma must be non-negative integers.")
    if L == 1:
        lags = [1]
    elif L == 2:
        lags = [1, int(seasonal_lag)]
    else:
        raise ValueError(
            "PTS supports at most one seasonal ARMA lag: ar/ma may have "
            f"length 1 (non-seasonal) or 2 (SARMA); got length {L}."
        )

    # Drop trailing zero blocks but always keep the non-seasonal (lag-1) block.
    keep = [i for i in range(L) if ar[i] > 0 or ma[i] > 0]
    if not keep:
        return {"ar": [0], "ma": [0], "lags": [1], "select": select}
    keep = sorted(set(keep) | {0})
    return {
        "ar": [ar[i] for i in keep],
        "ma": [ma[i] for i in keep],
        "lags": [lags[i] for i in keep],
        "select": select,
    }


def arma_spec(ar, ma, lags):
    """Build the (arma_orders tuple, irregular-option string) for a per-lag
    (ar, ma, lags) triple -- 2-tuple/`arma(p,q)` for non-seasonal, 5-tuple/
    `arma(p,q,P,Q,s)` for SARMA."""
    if len(lags) == 1:
        return (int(ar[0]), int(ma[0])), f"arma({int(ar[0])},{int(ma[0])})"
    tup = (int(ar[0]), int(ma[0]), int(ar[1]), int(ma[1]), int(lags[1]))
    return tup, f"arma({tup[0]},{tup[1]},{tup[2]},{tup[3]},{tup[4]})"


def ic_to_engine(ic: str) -> str:
    """Map adam-style ic (AICc/AIC/BIC/BICc) to the engine's lowercase string."""
    table = {"AICc": "aicc", "AIC": "aic", "BIC": "bic", "BICc": "bicc"}
    if ic not in table:
        raise ValueError(f"ic must be one of {list(table)}; got {ic!r}")
    return table[ic]
