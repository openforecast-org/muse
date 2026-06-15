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
#' @param lags seasonal period (default \code{frequency(data)}).
#' @param orders ARMA spec for the irregular component.  Two forms accepted:
#' \itemize{
#'   \item Full list \code{list(ar, ma, select)} — \code{ar} / \code{ma} are
#'     non-negative integers (default 0); \code{select = TRUE} asks the engine
#'     to search ARMA orders up to that cap (replaces the old \code{armaIdent}
#'     flag).
#'   \item Numeric shortcut \code{c(p, q)} — equivalent to
#'     \code{list(ar = p, ma = q, select = FALSE)}; \code{c(p)} is treated as
#'     \code{c(p, 0)}.
#' }
#' PTS has no differencing, so \code{orders$i} must be 0 if supplied.
#' @param formula optional formula \code{response ~ x1 + x2 + ...}; only
#' meaningful when \code{data} is a matrix or \code{data.frame}.  Used to
#' pick the response column + xreg columns explicitly.
#' @param regressors handling of xregs.  Currently only \code{"use"}
#' (apply all supplied xregs as fixed-coefficient covariates).  Adam's
#' \code{"select"} and \code{"adapt"} modes are not yet implemented.
#' @param ic information criterion for automatic model selection; one of
#' \code{"AICc"} (default), \code{"AIC"}, \code{"BIC"}, \code{"BICc"}.
#' Matches the adam option set.
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
                ic         = c("AICc", "AIC", "BIC", "BICc"),
                h          = 0,
                holdout    = FALSE,
                verbose    = FALSE,
                ...){
    cl  <- match.call()
    tic <- proc.time()
    regressors <- match.arg(regressors)
    ic         <- match.arg(ic)
    # Internal hatch (adam-style): if the caller passes B via ..., use it
    # as the starting parameter vector for the optimiser.  Natural-scale
    # (positive variances), matching the engine's userP0 branch.  Kept
    # out of the documented signature on purpose.
    dots <- list(...)
    B    <- dots$B
    criterion  <- .pts_ic_to_engine(ic)
    ordersUC   <- .pts_orders_to_uc(orders)
    if (!is.numeric(h) || length(h) != 1 || h < 0)
        stop("`h` must be a non-negative integer.", call. = FALSE)

    # Split the user-supplied `data` into the response vector y plus an
    # optional xreg matrix u.  Vector / ts / zoo go through unchanged;
    # matrix / data.frame either follow `formula` or default to "col 1 is
    # response, cols 2..k are xregs".
    parsed <- .pts_parse_data(data, formula = formula)
    y      <- parsed$y
    u      <- parsed$u

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

    res <- .pts_fit(y = y, u = u, model = model, lags = lags,
                    h = as.integer(h),
                    criterion = criterion,
                    armaIdent = ordersUC$select,
                    ar        = ordersUC$ar,
                    ma        = ordersUC$ma,
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
    # reports that in its $orders slot too.
    pq <- uc_to_arma(res$modelUC)
    ordersList <- list(ar = as.integer(pq[1]), i = 0L, ma = as.integer(pq[2]),
                       select = ordersUC$select)

    out <- list(
        ## --- inputs / spec ---
        # data: same wrapping convention as adam (.pts_wrap_in handles the
        # yClasses promotion + ts/zoo branch at adam.R:4489-4499).
        data       = .pts_wrap_in(y, y),
        u            = u,                # NULL when there are no regressors
        formula      = parsed$formula,
        responseName = parsed$responseName,
        regressors   = regressors,       # adam-aligned: "use" only for now
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
        # Matches greybox::alm at alm.R:2148 for distribution = "dbcnorm".
        nParam     = length(res$p) + as.integer(isTRUE(res$lambdaEstimated)),
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
        arma             = NULL,
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

#' @rdname pts
#' @description \code{auto.pts(data, ...)} is a thin wrapper that forces full
#'   automatic selection (\code{model = "ZZZ"}).
#' @export
auto.pts <- function(data, ...){
    pts(data, model = "ZZZ", ...)
}
