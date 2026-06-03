#' @title pts: Power / Trend / Seasonal state-space model
#'
#' @description Estimates a PTS (Power / Trend / Seasonal) state-space model
#' for a univariate time series.  This is the user-facing entry point of the
#' \pkg{muse} package and mirrors the calling convention used elsewhere in
#' the \pkg{smooth} family: \code{pts()} estimates the model, and
#' \code{\link{forecast.pts}} produces forecasts from the fitted object
#' without re-estimating.
#'
#' @param y univariate time series (numeric vector or \code{ts}).
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
#' @param lags seasonal period (default \code{frequency(y)}).
#' @param h forecast horizon. If \code{h > 0} a forecast is computed at fit
#' time and cached on the object; \code{forecast(object, h)} can later
#' recompute for a different horizon cheaply.
#' @param holdout logical. If \code{TRUE} and \code{h > 0}, the last \code{h}
#' observations of \code{y} are withheld from estimation and returned in
#' \code{$holdout} for later accuracy assessment.
#' @param criterion information criterion used for automatic model selection
#' (\code{"aic"}, \code{"bic"} or \code{"aicc"}).
#' @param armaIdent logical: search for an ARMA structure in the irregular
#' component.
#' @param verbose logical: print intermediate optimisation output.
#' @param u optional matrix of external regressors.
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
pts <- function(y, model = "ZZZ", lags = stats::frequency(y), h = 0,
                holdout = FALSE, criterion = c("aic", "bic", "aicc"),
                armaIdent = FALSE, verbose = FALSE, u = NULL){
    cl  <- match.call()
    tic <- proc.time()
    criterion <- match.arg(criterion)
    if (!is.numeric(h) || length(h) != 1 || h < 0)
        stop("`h` must be a non-negative integer.", call. = FALSE)

    held <- NULL
    if (holdout && h > 0){
        if (length(y) <= h)
            stop("`holdout = TRUE` requires `length(y) > h`.", call. = FALSE)
        n <- length(y) - h
        if (is.ts(y)){
            held <- stats::window(y, start = stats::time(y)[n + 1L])
            y    <- stats::window(y, end   = stats::time(y)[n])
        } else {
            held <- y[(n + 1L):length(y)]
            y    <- y[seq_len(n)]
        }
    }

    # The C++ engine always wants h >= 1; ask for one forecast point even
    # when the user asked for none, and drop the resulting cache below.
    h_fit <- max(as.integer(h), 1L)

    res <- .pts_fit(y = y, u = u, model = model, lags = lags, h = h_fit,
                    criterion = criterion, armaIdent = armaIdent,
                    verbose = verbose)

    cachedFor <- if (h > 0) res$yFor else NULL

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
    # with what the orders.pts accessor returns).
    pq <- uc_to_arma(res$modelUC)
    ordersList <- list(ar = as.integer(pq[1]), i = 0L, ma = as.integer(pq[2]))

    out <- list(
        ## --- inputs / spec ---
        # data: same wrapping convention as adam (.pts_wrap_in handles the
        # yClasses promotion + ts/zoo branch at adam.R:4489-4499).
        data       = .pts_wrap_in(y, y),
        model      = uc_to_pts(res$modelUC, res$lambda),
        modelUC    = res$modelUC,       # pts-specific UC string
        lags       = lags,
        lagsAll    = res$lagsAll,       # internal harmonic periods (C++ engine)
        lambda     = res$lambda,        # pts-specific Box-Cox parameter
        ## --- parameters ---
        B          = res$p,
        covp       = res$covp,          # vcov source (we have it directly; adam has $FI instead)
        nParam     = length(res$p),
        ## --- in-sample ---
        fitted     = res$fitted,        # original scale (back-transformed)
        residuals  = res$residuals,     # BC scale (engine innovations)
        comp       = res$comp,          # pts-specific BC-scale additive decomposition
        states     = statesMat,         # adam-aligned structural state evolution
        ## --- forecast cache (adam name = forecast; no variance slot) ---
        forecast      = cachedFor,
        forecast_args = res$forecast_args,
        ## --- likelihood + diagnostics ---
        logLik       = res$logLik,
        lossValue    = -as.numeric(res$logLik),  # adam: CFValue
        scale        = res$scale,                # MLE scale of dnorm
        table        = res$table,                # pts-specific C++ validation text
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
        formula          = NULL,
        regressors       = NA_character_,
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
#' @description \code{auto.pts(y, ...)} is a thin wrapper that forces full
#'   automatic selection (\code{model = "ZZZ"}).
#' @export
auto.pts <- function(y, ...){
    pts(y, model = "ZZZ", ...)
}
