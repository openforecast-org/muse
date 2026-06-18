#' @title pts: Power / Trend / Seasonal state-space model
#'
#' @description Estimates a PTS (Power / Trend / Seasonal) state-space model
#' for a univariate time series.  This is the user-facing entry point of the
#' \pkg{muse} package and mirrors the calling convention used elsewhere in
#' the \pkg{smooth} family: \code{pts()} estimates the model, and
#' \code{\link{forecast.pts}} produces forecasts from the fitted object
#' without re-estimating.
#'
#' @param data response series.  Either a univariate \code{ts} / \code{zoo} /
#' numeric vector, OR a matrix / \code{data.frame} whose first column is the
#' response and whose remaining columns are external regressors (xregs).
#' @param model 3-letter PTS specification string.  The three positions
#' encode Power / Trend / Seasonal:
#' \itemize{
#'   \item Power: \code{Z} to estimate Box-Cox \eqn{\lambda}, or a numeric
#'     value (e.g. \code{"0"}, \code{"0.5"}, \code{"1"}).
#'   \item Trend: \code{Z} (auto), \code{N} (none / random walk),
#'     \code{L} (local linear), \code{D} (damped / smooth random walk),
#'     \code{G} (global / deterministic).
#'   \item Seasonal: \code{Z} (auto), \code{N} (none), \code{D} (discrete /
#'     linear), \code{T} (trigonometric / equal).
#' }
#' @param lags seasonal period (default \code{frequency(data)}).  Scalar for
#' the structural seasonal; may also be a vector \code{c(1, s)} mirroring
#' \code{smooth::adam}'s convention — the last entry is the structural
#' period, the full vector becomes the default lag set for the irregular's
#' ARMA blocks (overridable per-fit via \code{orders$lags}).
#' @param orders ARMA / SARMA spec for the irregular component.  Three forms:
#' \itemize{
#'   \item Full list \code{list(ar, ma, select)} for a non-seasonal
#'     \eqn{ARMA(p, q)} — \code{ar}, \code{ma} non-negative scalars (default
#'     0); \code{select = TRUE} asks the engine to search ARMA orders up to
#'     that cap.
#'   \item Numeric shortcut \code{c(p, q)} — equivalent to
#'     \code{list(ar = p, ma = q, select = FALSE)}; \code{c(p)} is treated
#'     as \code{c(p, 0)}.
#'   \item Seasonal \code{list(ar = c(p, P), ma = c(q, Q), lags = c(1, s))}
#'     for \eqn{SARMA(p, q)(P, Q)_s} — \code{ar} / \code{ma} are length-L
#'     vectors paired position-wise with \code{lags}, with \code{lags[1] =
#'     1} (non-seasonal block) and \code{lags[2] = s} (seasonal block).
#'     If \code{orders$lags} is omitted the default falls back to the
#'     top-level \code{lags} argument (or \code{c(1, frequency(data))}).
#'     The seasonal SARMA polynomial is multiplied internally, so the BFGS
#'     only optimises the free \eqn{\phi_i, \Phi_j, \theta_i, \Theta_j}
#'     coefficients.  \code{select = TRUE} runs a grid search over every
#'     \eqn{(p', q', P', Q')} tuple with \eqn{0 \le p' \le} \code{ar[1]} and
#'     so on, and picks the candidate with the lowest \code{ic}.
#' }
#' PTS has no differencing — any \code{orders$i} supplied is silently
#' ignored.
#' @param formula optional formula \code{response ~ x1 + x2 + ...}; only
#' meaningful when \code{data} is a matrix or \code{data.frame}.  Used to
#' pick the response column + xreg columns explicitly.
#' @param regressors handling of xregs.  Currently only \code{"use"}
#' (apply all supplied xregs as fixed-coefficient covariates).  Adam's
#' \code{"select"} and \code{"adapt"} modes are not yet implemented.
#' @param outliers what to do about outliers, mirroring \code{adam()}'s
#' interface: \code{"ignore"} (default — fit ignores the possibility of
#' outliers) or \code{"use"} (run the engine's outlier detector once
#' after the structural fit, classify each event as AO / LS / SC, and
#' refit with the detected events as fixed regressor dummies).  Adam's
#' \code{"select"} mode (IC-pruning of detected dummies) is not yet
#' supported — passing it errors with a clear message.  When
#' \code{outliers = "use"} the detected events are returned on the
#' fitted object as \code{$outliersDetected} (a data frame with
#' \code{time} and \code{type} columns) and the corresponding dummy
#' coefficients appear in \code{coef(m)} as \code{Beta(...)} entries.
#' @param level confidence level driving the outlier z-score threshold
#' (default 0.99).  Translated to a two-sided z via
#' \code{qnorm((1 + level) / 2)}: 0.99 → ≈ 2.576, 0.95 → ≈ 1.960.  The
#' z drives the AO threshold; LS / SC scale proportionally to preserve
#' the engine's relative stiffness (LS ≈ 1.087×z, SC ≈ 1.304×z).
#' Ignored when \code{outliers = "ignore"}.
#' @param ic information criterion for automatic model selection; one of
#' \code{"BICc"} (default), \code{"AICc"}, \code{"BIC"}, \code{"AIC"}.
#' Matches the adam option set.  BICc is the default because its stiffer
#' complexity penalty guards against runaway-trend candidates whose
#' likelihood beats simpler shapes by only a couple of units.
#' @param h forecast horizon. If \code{h > 0} a forecast is computed at fit
#' time and cached on the object; \code{forecast(object, h)} can later
#' recompute for a different horizon cheaply.
#' @param holdout logical. If \code{TRUE} and \code{h > 0}, the last \code{h}
#' observations of \code{data} are withheld from estimation and returned in
#' \code{$holdout} for later accuracy assessment.
#' @param verbose logical: print intermediate optimisation output.
#' @param ... advanced / undocumented passthroughs.  Supported keys:
#' \itemize{
#'   \item \code{B} - numeric vector of starting values for the optimiser
#'     (natural-scale variances, in the order returned in \code{$B} by a
#'     default fit).  Mirrors the same hatch in \code{smooth::adam()}.
#'     The optimised vector is returned in the \code{$B} slot of the
#'     output regardless of whether the user supplied one.
#' }
#'
#' @return An object of class \code{c("pts", "smooth")}.  Slot names mirror
#' \code{smooth::adam()}'s return list where the concept is shared; pts-only
#' extensions are flagged below.
#' \itemize{
#'   \item Inputs / spec: \code{y, model, modelUC*, lags, lambda*}
#'   \item Parameters: \code{B} (estimated parameter vector), \code{covp*}
#'     (parameter covariance), \code{nParam}
#'   \item In-sample fit: \code{fitted, residuals, states} plus
#'     pts-specific \code{comp*} (additive BC-scale decomposition with
#'     Error/Fit columns)
#'   \item Cached forecast: \code{forecast} (original scale, if \code{h > 0})
#'     and \code{forecast_args*} for cheap re-forecasting
#'   \item Likelihood + diagnostics: \code{logLik},
#'     \code{table*} (C++ validation text)
#'   \item Scalars read by \code{plot.smooth} / diagnostics:
#'     \code{distribution = "dnorm"}, \code{loss = "likelihood"},
#'     \code{occurrence = NULL}, \code{holdout}
#'   \item Bookkeeping: \code{call, timeElapsed}
#' }
#' AIC / AICc / BIC / BICc are derived on demand via the methods, not
#' stored on the object.  (* = pts-specific extension.)
#'
#' @seealso \code{\link{forecast.pts}}
#'
#' @template authors
#' @export
pts <- function(data,
                model      = "ZZZ",
                lags       = stats::frequency(data),
                orders     = list(ar = 0, ma = 0, select = FALSE),
                formula    = NULL,
                regressors = c("use"),
                outliers   = c("ignore", "use", "select"),
                level      = 0.99,
                ic         = c("BICc", "AICc", "BIC", "AIC"),
                h          = 0,
                holdout    = FALSE,
                verbose    = FALSE,
                ...){
    cl  <- match.call()
    tic <- proc.time()
    regressors <- match.arg(regressors)
    ic         <- match.arg(ic)
    outliers   <- match.arg(outliers)
    if (outliers == "select")
        stop("`outliers = 'select'` is not yet supported; use 'use'.",
             call. = FALSE)
    if (!is.numeric(level) || length(level) != 1L ||
        level <= 0 || level >= 1)
        stop("`level` must be a length-1 numeric in (0, 1).",
             call. = FALSE)
    # Engine takes a z-score threshold (positive → enable outlier
    # detection); adam-style `level` converts to a two-sided z via the
    # standard normal quantile at (1 + level)/2.  At the default 0.99 →
    # ≈ 2.576 ≈ qnorm(0.995, 0, 1).
    outlier_z <- if (outliers == "ignore") 0
                 else stats::qnorm((1 + level) / 2)
    # Internal hatch (adam-style): if the caller passes B via ..., use it
    # as the starting parameter vector for the optimiser.  Natural-scale
    # (positive variances), matching the engine's userP0 branch.  Kept
    # out of the documented signature on purpose.
    dots <- list(...)
    B    <- dots$B
    criterion  <- .pts_ic_to_engine(ic)
    # `lags` historically is the scalar structural-seasonal period.  As an
    # adam-style ergonomic, also accept a vector c(1, s, ...) — the last
    # entry is taken as the structural period; the full vector becomes the
    # default ARMA-block lag set when `orders$lags` is not supplied.
    if (length(lags) > 1L){
        lagsDefault <- as.integer(lags)
        lags        <- as.integer(lags[length(lags)])
    } else {
        lagsDefault <- c(1L, as.integer(lags))
    }
    ordersUC <- .pts_orders_to_uc(orders, lagsDefault = lagsDefault)
    if (!is.numeric(h) || length(h) != 1 || h < 0)
        stop("`h` must be a non-negative integer.", call. = FALSE)

    # Split the user-supplied `data` into the response vector y plus an
    # optional xreg matrix u.  Vector / ts / zoo go through unchanged;
    # matrix / data.frame either follow `formula` or default to "col 1 is
    # response, cols 2..k are xregs".
    parsed <- .pts_parse_data(data, formula = formula)
    y      <- parsed$y
    u      <- parsed$u

    # Lower bound on lambda for the engine's joint-BFGS:
    #   * any(y < 0)  -> lambdaLower = 0     (BC undefined for negatives)
    #   * any(y == 0) -> lambdaLower = 1e-10 (BC at lambda <= 0 is +/-Inf)
    # Otherwise unbounded.  The bound is enforced inside `llik()` in the
    # C++ engine; no R-side rewrites of the model spec.
    lambdaLower <- if (any(y < 0, na.rm = TRUE)) 0
                   else if (any(y == 0, na.rm = TRUE)) 1e-10
                   else -Inf
    # Negative-y safety: Box-Cox throws for any lambda != 1 when y has
    # negatives, so warn the user and pin the spec to lambda = 1 by
    # rewriting position 1 of the model string.  An explicit lambda = 1
    # is left untouched and stays silent.
    if (any(y < 0, na.rm = TRUE)){
        nm <- nchar(model)
        lambdaUser <- suppressWarnings(as.numeric(substr(model, 1L, nm - 2L)))
        if (is.na(lambdaUser) || lambdaUser != 1){
            warning("`data` contains negative values; Box-Cox is undefined for ",
                    "negatives.  Setting lambda = 1 (no transformation).",
                    call. = FALSE)
            model <- paste0("1", substr(model, nm - 1L, nm))
        }
    }

    held <- NULL
    if (holdout && h > 0){
        if (length(y) <= h)
            stop("`holdout = TRUE` requires `length(data) > h`.", call. = FALSE)
        n <- length(y) - h
        if (is.ts(y)){
            held <- stats::window(y, start = stats::time(y)[n + 1L])
            y    <- stats::window(y, end   = stats::time(y)[n])
        } else if (inherits(y, "zoo")){
            held <- y[(n + 1L):length(y)]
            y    <- y[seq_len(n)]
        } else {
            held <- y[(n + 1L):length(y)]
            y    <- y[seq_len(n)]
        }
        if (!is.null(u)){
            # u is k x N; split column-wise to keep the kxn / kxh shapes.
            u_held <- u[, (n + 1L):ncol(u), drop = FALSE]
            u      <- u[, seq_len(n),       drop = FALSE]
        }
    }

    # Box-Cox lambda screening: when the user requests auto-lambda ("Z"),
    # use Brent's method to minimise the AIC of a fixed-structure proxy
    # model (local linear trend + trigonometric seasonal -- LN when the
    # series is non-seasonal) over the continuous lambda range
    # [lambdaLower, LAMBDA_BOUND_UPPER].  The "Z" lambda slot is then
    # rewritten with the resulting numeric value, so the downstream
    # structural ident runs at fixed lambda (no joint-BFGS for lambda).
    #
    # On the eurotourism benchmark (325 monthly series) Brent gives the
    # best aggregate AIC of any approach measured (-7700 AIC vs the
    # joint-BFGS + snap path), substantially better than the previous
    # 8-anchor LT screen (-895 vs joint+snap).  An earlier variant that
    # also tested each engine snap anchor on top of Brent (8 extra LT
    # fits per series) added 30% runtime for ~0 AIC benefit -- Brent
    # already converges to the continuous AIC minimum, and the snap
    # anchors aren't load-bearing once lambda is fixed before entering
    # the engine.
    #
    # nParam is incremented by one when Brent runs, since lambda was
    # then effectively estimated from data (Brent does cost ~1 DoF
    # despite the engine running with a numerically-fixed lambda).
    lambdaWasScreened <- FALSE
    nm <- nchar(model)
    if (toupper(substr(model, 1L, nm - 2L)) == "Z" && length(y) >= 4){
        seas_letter <- if (length(lags) > 0L &&
                           utils::tail(as.integer(lags), 1L) > 1L) "T" else "N"
        ltAIC <- function(lam){
            spec_lt <- paste0(format(lam, scientific = FALSE,
                                     drop0trailing = TRUE),
                              "L", seas_letter)
            m_lt <- tryCatch(
                suppressWarnings(pts(y, model = spec_lt, lags = lags, h = 0)),
                error = function(e) NULL)
            if (is.null(m_lt)) return(Inf)
            a <- stats::AIC(m_lt)
            if (!is.finite(a)) Inf else a
        }
        lower <- if (is.finite(lambdaLower)) max(lambdaLower, -2) else -2
        upper <- 2
        opt <- tryCatch(stats::optimize(ltAIC, lower = lower, upper = upper,
                                        tol = 1e-3),
                        error = function(e) NULL)
        bestLambda <- if (!is.null(opt) && is.finite(opt$objective))
                          opt$minimum else 1
        model <- paste0(format(bestLambda, scientific = FALSE,
                               drop0trailing = TRUE),
                        substr(model, nm - 1L, nm))
        lambdaWasScreened <- TRUE
    }

    # Nested PTS + ARMA selection.  When `orders$select = TRUE`, the loop
    # is PTS-outer / ARMA-inner: for every structural (trend, seasonal)
    # candidate, fit it without ARMA, run the ARMA grid on its residuals,
    # and score combined_IC = structural_IC + (best_ARMA_IC − ARMA(0,0)_IC).
    # The (PTS structure, ARMA orders) pair with the lowest combined IC
    # wins; the final fit then runs at fixed structure + fixed ARMA.
    #
    # The damped (D / srw) trend is paired only with AR-less ARMA
    # candidates because srw's α parameter and AR persistence are
    # jointly unidentified — same rule the engine's findUCmodels() applies.
    userSelect <- isTRUE(ordersUC$select)
    if (userSelect){
        sel <- .pts_select_pts_arma(y = y, u = u,
                                    model_template = model,
                                    lags     = lags,
                                    ar_max   = ordersUC$ar,
                                    ma_max   = ordersUC$ma,
                                    arma_lags = ordersUC$lags,
                                    ic        = ic,
                                    criterion = criterion,
                                    verbose   = verbose)
        # Lock the structural spec down (lambda + trend + seasonal letters)
        # so the final fit doesn't re-ident.
        model <- sel$model_spec
        ordersUC$ar     <- sel$ar
        ordersUC$ma     <- sel$ma
        ordersUC$lags   <- sel$lags
        ordersUC$select <- FALSE
    }

    # The engine's outlier-injection path has a parameter-vector
    # dimensionality bug when joint-BFGS is also estimating lambda (the
    # extra lambda slot doesn't survive the dummy-injection refit).
    # Sidestep by pinning lambda *before* outlier detection: when
    # `outliers = "use"` and the model spec still has the auto-lambda
    # `Z` letter, run a quick no-outlier fit, read its chosen lambda
    # back, and rewrite the model spec with that lambda as a number.
    if (outliers == "use" && outlier_z > 0){
        nmod <- nchar(model)
        if (toupper(substr(model, 1L, nmod - 2L)) == "Z"){
            pre <- .pts_fit(y = y, u = u, model = model, lags = lags,
                            h = 0L,
                            criterion = criterion,
                            armaIdent = FALSE,
                            ar = ordersUC$ar, ma = ordersUC$ma,
                            armaLags = ordersUC$lags,
                            outlier = 0,
                            B = NULL, verbose = FALSE)
            lambda_chosen <- round(as.numeric(pre$lambda), 4)
            model <- paste0(format(lambda_chosen, scientific = FALSE),
                            substr(model, nmod - 1L, nmod))
        }
    }

    res <- .pts_fit(y = y, u = u, model = model, lags = lags,
                    h = as.integer(h),
                    criterion = criterion,
                    armaIdent = ordersUC$select,
                    ar        = ordersUC$ar,
                    ma        = ordersUC$ma,
                    armaLags  = ordersUC$lags,
                    outlier   = outlier_z,
                    lambdaLower = lambdaLower,
                    B         = B,
                    verbose   = verbose)
    # When h > 0 we cache the engine's forecast (length h, original scale).
    # When h == 0 we still populate $forecast with a 1-period NA placeholder
    # anchored at the next observation, mirroring adam.R:572:
    #   ts(NA, start = yIndex[obsInSample] + diff(yIndex[1:2]),
    #      frequency = yFrequency)
    cachedFor <- if (h > 0) res$yFor else .pts_wrap_oos(NA_real_, y)

    # Structural state evolution: build the (nobs + 1) x nStates matrix
    # (row 1 = initial state at t = 0, rows 2..n+1 = smoothed states), then
    # let .pts_wrap_states attach the right ts/zoo time class with the
    # leading row anchored at start(data) - one period (adam convention).
    statesMat <- NULL
    if (is.matrix(res$comp) && ncol(res$comp) >= 3){
        ns   <- length(y)
        cols <- setdiff(colnames(res$comp), c("Error", "Fit"))
        raw  <- res$comp[, cols, drop = FALSE]
        if (nrow(raw) > ns) raw <- raw[seq_len(ns), , drop = FALSE]
        statesMat <- rbind(NA_real_, unclass(raw))
        colnames(statesMat) <- colnames(raw)
        statesMat <- .pts_wrap_states(statesMat, y)
    }

    # ARMA orders from the UC string (derived once so $orders is consistent
    # with what the orders.pts accessor returns).  We carry the user's
    # `select` flag through so a model fitted with orders$select = TRUE
    # reports that in its $orders slot too.  `pq` carries per-lag vectors so
    # SARMA specs round-trip cleanly.
    pq <- uc_to_arma(res$modelUC)
    ordersList <- list(ar     = as.integer(pq$ar),
                       ma     = as.integer(pq$ma),
                       lags   = as.integer(pq$lags),
                       # Preserve the user's original select flag so a fit
                       # done via the residual-based grid search still
                       # reports `$orders$select = TRUE`.
                       select = userSelect)

    # Flat AR / MA coefficient vectors (smooth::adam convention — `$arma`).
    # Lag-by-lag concatenation: c(φ_1, ..., φ_p [, Φ_1, ..., Φ_P]) for AR
    # and analogously for MA.  Empty when the fit has no ARMA structure.
    Bnames  <- names(res$p)
    armaList <- list(
        ar = unname(res$p[grepl("^S?AR\\(", Bnames)]),
        ma = unname(res$p[grepl("^S?MA\\(", Bnames)])
    )

    out <- list(
        ## --- inputs / spec ---
        # data: same wrapping convention as adam (.pts_wrap_in handles the
        # yClasses promotion + ts/zoo branch at adam.R:4489-4499).
        data       = .pts_wrap_in(y, y),
        u            = u,                # NULL when there are no regressors
        formula      = parsed$formula,
        responseName = parsed$responseName,
        regressors   = regressors,       # adam-aligned: "use" only for now
        outliers     = outliers,         # adam-aligned: "ignore" or "use"
        level        = level,            # confidence level for outlier z-threshold
        outliersDetected = res$outliersDetected,
        ic           = ic,               # adam-style criterion name (AICc/...)
        model      = uc_to_pts(res$modelUC, res$lambda),
        modelUC    = res$modelUC,       # pts-specific UC string
        lags       = lags,
        lagsAll    = res$lagsAll,       # internal harmonic periods (C++ engine)
        lambda     = res$lambda,        # pts-specific Box-Cox parameter
        ## --- parameters ---
        B          = res$p,
        vcov       = res$covp,          # parameter covariance, computed by the
                                        # C++ "all" command at no extra cost
        # Count the Box-Cox lambda as one additional DoF when the user
        # asked the engine to estimate it (model spec started with "Z").
        # Two paths can produce an estimated lambda: (a) the engine's
        # joint-BFGS path returning lambdaEstimated = TRUE; (b) the
        # R-side Brent screen above replacing "Z" with a numeric value
        # before the structural ident runs.  Either way costs +1 DoF.
        # Matches greybox::alm at alm.R:2148 for distribution = "dbcnorm".
        # G (deterministic) trend: drift slope is concentrated out as a
        # regressor (PTSmodel.h:3246), so parLabels() does not push a
        # "Slope" entry and length(res$p) misses it.  Still an estimated
        # DoF — add it back here.
        nParam     = length(res$p) +
                     as.integer(isTRUE(res$lambdaEstimated) ||
                                lambdaWasScreened) +
                     as.integer(grepl("^td/", res$modelUC)),
        ## --- in-sample ---
        fitted     = res$fitted,        # original scale (back-transformed)
        residuals  = res$residuals,     # BC scale (engine innovations)
        comp       = res$comp,          # pts-specific BC-scale additive decomposition
        states     = statesMat,         # adam-aligned structural state evolution
        ## --- forecast convenience cache (NULL when pts is called with h = 0) ---
        forecast     = cachedFor,
        ## --- likelihood + diagnostics ---
        logLik       = res$logLik,
        lossValue    = -as.numeric(res$logLik),  # adam: CFValue
        scale        = res$scale,                # MLE scale of dnorm
        cppOutput    = res$table,                # raw C++ validation text block
        ## --- smooth/adam-aligned scalars for plot.smooth dispatch ---
        distribution = "dnorm",
        loss         = "likelihood",
        lossFunction = NULL,
        occurrence   = NULL,                     # is.occurrence(NULL) == FALSE
        holdout      = NULL,                     # overwritten below if holdout = TRUE
        ## --- adam-aligned slots that PTS has no analog for; values
        ## mirror what adam stores when the corresponding feature is
        ## absent (smooth/R/adam.R:578-612).  NA for atomic; NULL for
        ## list-typed.  Keeping them in the return list keeps `names(m)`
        ## in line with adam so downstream tooling can introspect by name. ---
        persistence      = NA_real_,
        phi              = NA_real_,
        transition       = NA,
        measurement      = NA,
        initial          = NA,
        initialType      = NA_character_,
        initialEstimated = NA,
        orders           = ordersList,
        arma             = armaList,
        constant         = NA_real_,
        other            = NULL,
        ets              = NA,
        res              = NA,
        FI               = NA,
        adamCpp          = NA,
        profile          = NULL,
        profileInitial   = NULL,
        ## --- bookkeeping ---
        call         = cl,
        timeElapsed  = proc.time() - tic
    )
    if (!is.null(held)) out$holdout <- held
    class(out) <- c("pts", "smooth")
    out
}
