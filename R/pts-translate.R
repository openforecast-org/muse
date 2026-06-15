# Internal translators between PTS 3-letter spec and the UC model string
# that the C++ engine speaks. Not exported.

# pts_to_uc: turn a PTS spec ("0NT", "ZZZ", "0.5LD") into a UC string and
# a Box-Cox lambda.
#
# armaOrders specifies the irregular ARMA structure on the UC side when
# armaSelect = FALSE:
#   * c(p, q)        — non-seasonal ARMA(p,q), serialised as "arma(p,q)";
#   * c(p, q, P, Q, s) — SARMA(p,q)(P,Q)_s, serialised as
#     "arma(p,q,P,Q,s)".  The engine recognises both grammars by counting
#     commas inside the parens (2 → non-seasonal, 5 → seasonal).
# When armaSelect = TRUE the irregular slot becomes the engine's "?" sentinel
# so ident() searches over the candidate list passed in irregularOptions
# (see .pts_arma_candidates).
pts_to_uc <- function(model, armaOrders = c(0, 0), armaSelect = FALSE){
    out <- list(modelU = "", lambda = 1.0)
    n   <- nchar(model)
    # ARMA / SARMA irregular component
    if (isTRUE(armaSelect)) {
        modelU <- "/?"
    } else if (length(armaOrders) == 2L) {
        modelU <- paste0("/arma(", armaOrders[1], ",", armaOrders[2], ")")
    } else if (length(armaOrders) == 5L) {
        modelU <- paste0("/arma(", armaOrders[1], ",", armaOrders[2], ",",
                         armaOrders[3], ",", armaOrders[4], ",",
                         armaOrders[5], ")")
    } else {
        stop("Internal error: armaOrders must have length 2 (arma) or 5 ",
             "(sarma).", call. = FALSE)
    }
    # Seasonal
    aux <- tolower(substr(model, n, n))
    if      (aux == "z") modelU <- paste0("/?",     modelU)
    else if (aux == "n") modelU <- paste0("/none",  modelU)
    else if (aux == "d") modelU <- paste0("/linear",modelU)
    else if (aux == "t") modelU <- paste0("/equal", modelU)
    else stop("Invalid seasonal letter in PTS spec: '", aux, "'", call. = FALSE)
    # Trend
    aux <- tolower(substr(model, n - 1, n - 1))
    if      (aux == "z") modelU <- paste0("?/none",  modelU)
    else if (aux == "n") modelU <- paste0("rw/none", modelU)
    else if (aux == "l") modelU <- paste0("llt/none",modelU)
    else if (aux == "d") modelU <- paste0("srw/none",modelU)
    else if (aux == "g") modelU <- paste0("td/none", modelU)
    else stop("Invalid trend letter in PTS spec: '", aux, "'", call. = FALSE)
    out$modelU <- modelU
    # Power (Box-Cox lambda): either numeric or 'Z' for estimate
    aux <- tolower(substr(model, 1, n - 2))
    num <- suppressWarnings(as.numeric(aux))
    if (!is.na(num))      out$lambda <- num
    else if (aux == "z")  out$lambda <- 9999.9   # C++ sentinel for "estimate"
    else stop("Invalid power letter in PTS spec: '", aux, "'", call. = FALSE)
    out
}

# uc_to_pts: format a fully-resolved UC string + lambda as
#   PTS(<lambda>,<trend letter>,<seasonal letter>)
# e.g. PTS(0,N,T), PTS(-0.29,D,T), PTS(1,L,N).  Unknown trend / seasonal
# tokens collapse to an empty field rather than silently dropping a slot.
uc_to_pts <- function(modelUC, lambda){
    modelUC <- sub("/none/", "/", modelUC, fixed = TRUE)
    sl <- gregexpr("/", modelUC)[[1]]
    trend    <- substr(modelUC, 1,         sl[1] - 1)
    seasonal <- substr(modelUC, sl[1] + 1, sl[2] - 1)
    trendLetter <- switch(trend,
                          "rw"  = "N",
                          "srw" = "D",
                          "llt" = "L",
                          "td"  = "G",
                          "?"   = "Z",
                          "")
    seasonalLetter <- switch(seasonal,
                             "none"   = "N",
                             "equal"  = "T",
                             "linear" = "D",
                             "?"      = "Z",
                             "")
    sprintf("PTS(%s,%s,%s)",
            as.character(round(lambda, 2)),
            trendLetter,
            seasonalLetter)
}

# .pts_orders_to_uc: turn adam-style orders + lags into the per-lag (ar, ma)
# vectors + select flag that the UC engine consumes.  Accepts:
#
#   * list(ar, ma, select) — the canonical adam form.  `ar` and `ma` may be
#     scalars (non-seasonal) or length-L vectors paired position-wise with
#     `lags` for SARMA(p,q)(P,Q)_s and so on.
#   * c(p, q) numeric vector — shorthand for list(ar = p, ma = q,
#     select = FALSE).  A scalar c(p) is treated as c(p, 0).
#
# PTS has no differencing (engine has no `i` flow) and ARMA only sits on the
# irregular component, so we validate i == 0 (when supplied) and reject
# negatives.
#
# `lagsDefault` provides the implicit seasonal lag when `orders$lags` is not
# supplied but `ar` / `ma` are vectors (typically frequency(data)).
.pts_orders_to_uc <- function(orders, lagsDefault = 1L){
    if (is.null(orders)) orders <- list()
    # Vector shorthand: c(p) or c(p, q) → list(ar = p, ma = q, select = FALSE)
    if (is.numeric(orders) && !is.list(orders)){
        if (length(orders) < 1 || length(orders) > 2)
            stop("Numeric `orders` shortcut must be c(p) or c(p, q); ",
                 "use list(ar = ..., ma = ..., select = ...) for full control.",
                 call. = FALSE)
        orders <- list(ar = orders[1],
                       ma = if (length(orders) == 2) orders[2] else 0L,
                       select = FALSE)
    }
    # PTS has no differencing — silently drop any `orders$i` the caller
    # supplies (typical when reusing an adam-style spec).
    orders$i <- NULL
    ar <- if (is.null(orders$ar)) 0L else as.integer(orders$ar)
    ma <- if (is.null(orders$ma)) 0L else as.integer(orders$ma)
    sel <- isTRUE(orders$select)
    # Pad ar / ma to common length so vector arithmetic with `lags` is clean.
    L  <- max(length(ar), length(ma), 1L)
    if (length(ar) < L) ar <- c(ar, rep(0L, L - length(ar)))
    if (length(ma) < L) ma <- c(ma, rep(0L, L - length(ma)))
    # Lag vector: explicit > orders$lags > default (1 for non-seasonal,
    # c(1, frequency(data)) for seasonal — passed in as lagsDefault).
    lags <- if (!is.null(orders$lags)) as.integer(orders$lags) else NULL
    if (is.null(lags)){
        if (L == 1L) lags <- 1L
        else if (L == length(lagsDefault)) lags <- as.integer(lagsDefault)
        else stop("Vector `orders$ar` / `orders$ma` of length ", L,
                  " requires `lags` of the same length.", call. = FALSE)
    }
    if (length(lags) != L)
        stop("`lags` length (", length(lags), ") must match `orders$ar` / ",
             "`orders$ma` length (", L, ").", call. = FALSE)
    if (any(ar < 0) || any(ma < 0))
        stop("`orders$ar` and `orders$ma` must be non-negative integers.",
             call. = FALSE)
    if (any(lags < 1L))
        stop("`lags` entries must be >= 1.", call. = FALSE)
    # Drop redundant trailing zero-order blocks so the UC encoding is minimal.
    keep <- ar > 0L | ma > 0L
    if (!any(keep)){
        # Pure noise: emit a single non-seasonal arma(0,0) block.
        ar <- 0L; ma <- 0L; lags <- 1L
    } else {
        # Keep the first lag (non-seasonal) plus any block with non-zero
        # orders so the encoding is unambiguous.
        keep[1L] <- TRUE
        ar <- ar[keep]; ma <- ma[keep]; lags <- lags[keep]
    }
    if (length(lags) >= 2L && any(duplicated(lags)))
        stop("`lags` entries must be unique.", call. = FALSE)
    if (length(lags) >= 2L && lags[1L] != 1L)
        stop("`lags[1]` must equal 1 (the non-seasonal block); seasonal ",
             "lags go in lags[2:L].", call. = FALSE)
    if (length(lags) > 2L)
        stop("PTS currently supports at most one seasonal lag in the ",
             "irregular ARMA — got length(lags) = ", length(lags),
             ".  Combine multiple seasonal lags into a single ",
             "SARMA(p,q)(P,Q)_s before passing in.", call. = FALSE)
    list(ar = ar, ma = ma, lags = lags, select = sel)
}

# .pts_ic_to_engine: map adam-style ic (AICc / AIC / BIC / BICc) to the
# engine's lowercase criterion string.
.pts_ic_to_engine <- function(ic){
    ic <- match.arg(ic, c("AICc", "AIC", "BIC", "BICc"))
    switch(ic,
           "AICc" = "aicc",
           "AIC"  = "aic",
           "BIC"  = "bic",
           "BICc" = "bicc")
}

# .pts_arma_candidates: serialise a single fixed ARMA spec for the engine's
# irregularOptions slot.  Format:
#   length-1 lags -> "arma(p,q)"
#   length-2 lags -> "arma(p,q,P,Q,s)"  (SARMA(p,q)(P,Q)_s)
#
# Order selection is now done R-side in .pts_select_arma() before the call
# to .pts_fit, so the engine never sees a multi-entry candidate list from
# pts().  The `select` parameter is accepted only for backwards-compatible
# call-site shape; it is ignored.
.pts_arma_candidates <- function(ar, ma, lags = 1L, select = FALSE){
    ar <- as.integer(ar); ma <- as.integer(ma); lags <- as.integer(lags)
    if (length(lags) == 1L)
        return(sprintf("arma(%d,%d)", ar[1L], ma[1L]))
    sprintf("arma(%d,%d,%d,%d,%d)",
            ar[1L], ma[1L], ar[2L], ma[2L], lags[2L])
}

# .pts_select_arma: residual-based ARMA grid search.  Every candidate
# (including the no-ARMA (0, 0) baseline) goes through the engine's own
# `UCompARMAC` so the IC ranking is computed by a single likelihood
# definition — the same MLE σ̂², the same PACF parameter space, and the
# same DoF counting (k = 1 + sum(arOrders) + sum(maOrders)) as the final
# PTS+ARMA fit will use.  No stats::arima call anywhere; muse relies on
# its own implementation end-to-end.
#
# `residuals` is the BC-scale residual vector from a structural-only PTS
# fit.  `lags` is length 1 (non-seasonal) or 2 (seasonal with seasonal
# period in lags[2]).  Returns list(ar, ma, lags) with the winning per-lag
# vectors.
.pts_select_arma <- function(residuals, ar_max, ma_max, lags, ic = "AICc",
                             verbose = FALSE){
    ar_max <- as.integer(ar_max); ma_max <- as.integer(ma_max)
    lags   <- as.integer(lags)
    ic     <- match.arg(ic, c("AICc", "AIC", "BIC", "BICc"))
    r      <- as.numeric(residuals)
    r      <- r[is.finite(r)]
    n      <- length(r)
    if (n < 4L)
        stop("Not enough finite residuals (", n, ") to fit ARMA candidates.",
             call. = FALSE)
    seasonal <- (length(lags) == 2L)
    grid <- if (!seasonal)
        expand.grid(p = 0L:ar_max[1L], q = 0L:ma_max[1L],
                    P = 0L, Q = 0L,
                    KEEP.OUT.ATTRS = FALSE)
    else
        expand.grid(p = 0L:ar_max[1L], q = 0L:ma_max[1L],
                    P = 0L:ar_max[2L], Q = 0L:ma_max[2L],
                    KEEP.OUT.ATTRS = FALSE)

    # Verbose header — matches the formatting of the engine's PTS ident
    # table (PTSmodel.h around line 1867) so the two grid traces sit
    # nicely on top of one another in the console.
    bar <- paste(rep("-", 83), collapse = "")
    if (isTRUE(verbose)){
        cat(bar, "\n")
        cat(" ARMA grid search on structural residuals:\n")
        cat(bar, "\n")
        cat(sprintf(" %-22s %13s %13s %13s %13s\n",
                    "   Model", "AIC", "AICc", "BIC", "BICc"))
        cat(bar, "\n")
    }
    label_one <- function(p, q, P, Q){
        if (!seasonal) sprintf("arma(%d,%d)", p, q)
        else           sprintf("arma(%d,%d,%d,%d,%d)", p, q, P, Q, lags[2L])
    }
    icKey <- ic   # one of {AIC, AICc, BIC, BICc}
    score_one <- function(p, q, P, Q){
        if (!seasonal){
            arOrders <- as.integer(p)
            maOrders <- as.integer(q)
            armaLagsV <- 1L
        } else {
            arOrders <- as.integer(c(p, P))
            maOrders <- as.integer(c(q, Q))
            armaLagsV <- as.integer(lags)
        }
        res <- UCompARMAC(r, arOrders, maOrders, armaLagsV, "aic")
        if (isTRUE(verbose)){
            if (isTRUE(res$succeed)){
                cat(sprintf(" %-22s %13.4f %13.4f %13.4f %13.4f\n",
                            label_one(p, q, P, Q),
                            res$AIC, res$AICc, res$BIC, res$BICc))
            } else {
                cat(sprintf(" %-22s   %s\n",
                            label_one(p, q, P, Q), "failed"))
            }
        }
        if (!isTRUE(res$succeed)) return(Inf)
        val <- res[[icKey]]
        if (is.null(val) || !is.finite(val)) Inf else val
    }
    ics <- mapply(score_one, grid$p, grid$q, grid$P, grid$Q)
    best <- which.min(ics)
    if (!length(best) || !is.finite(ics[best])){
        best_row <- list(p = 0L, q = 0L, P = 0L, Q = 0L)
    } else {
        best_row <- as.list(grid[best, ])
    }
    if (isTRUE(verbose)){
        cat(bar, "\n")
        cat(sprintf(" Selected by %s: %s\n", ic,
                    label_one(best_row$p, best_row$q,
                              best_row$P, best_row$Q)))
        cat(bar, "\n")
    }
    if (!seasonal)
        list(ar = as.integer(best_row$p), ma = as.integer(best_row$q),
             lags = 1L)
    else
        list(ar = as.integer(c(best_row$p, best_row$P)),
             ma = as.integer(c(best_row$q, best_row$Q)),
             lags = lags)
}

# uc_to_arma: pull the ARMA orders out of an "arma(...)" block embedded in a
# UC string.  Returns a list with $ar, $ma (vectors) and $lags (vector) so the
# caller can round-trip seasonal specs.  Empty / missing arma → c(0, 0) at
# lag 1.
uc_to_arma <- function(model){
    p1 <- regexpr("arma\\(", model)
    if (p1 < 1)
        return(list(ar = 0L, ma = 0L, lags = 1L))
    p2 <- regexpr("\\)", substring(model, p1))
    body <- substring(model, p1 + 5L, p1 + p2 - 2L)
    parts <- suppressWarnings(as.integer(strsplit(body, ",",
                                                  fixed = TRUE)[[1L]]))
    parts[is.na(parts)] <- 0L
    if (length(parts) <= 2L){
        ar   <- parts[1L]; if (is.na(ar)) ar <- 0L
        ma   <- if (length(parts) >= 2L) parts[2L] else 0L
        list(ar = as.integer(ar), ma = as.integer(ma), lags = 1L)
    } else if (length(parts) == 5L){
        list(ar = as.integer(c(parts[1L], parts[3L])),
             ma = as.integer(c(parts[2L], parts[4L])),
             lags = as.integer(c(1L, parts[5L])))
    } else {
        list(ar = 0L, ma = 0L, lags = 1L)
    }
}
