# Internal helpers used by pts() and forecast.pts(). Not exported.

# .pts_uc_inputs: marshal y / u / model / horizon / options into the exact
# argument list the C++ UCompC entry point expects. Returns a named list
# that we pass to UCompC and also stash on the pts object for forecast.pts.
.pts_uc_inputs <- function(y, u, modelUC, h, lambda, criterion, lags,
                           verbose, armaIdent){
    # u: NULL -> sentinel matrix; vector -> row matrix; otherwise pass through
    if (is.null(u)){
        u_mat <- matrix(0, 1, 2)
    } else if (is.null(dim(u))){
        u_mat <- matrix(u, 1, length(u))
    } else {
        u_mat <- u
        if (nrow(u_mat) > ncol(u_mat)) u_mat <- t(u_mat)
    }
    # periods derived from seasonal lag (harmonic decomposition)
    periods <- lags / (1 : max(1L, floor(lags / 2L)))
    rhos    <- rep(1, length(periods))
    list(
        y                = y,
        u                = u_mat,
        model            = modelUC,
        h                = as.integer(h),
        lambda           = lambda,
        outlier          = 0,
        tTest            = FALSE,
        criterion        = criterion,
        periods          = periods,
        rhos             = rhos,
        verbose          = verbose,
        stepwise         = FALSE,
        p                = -9999.9,
        arma             = armaIdent,
        TVP              = -9999.99,
        seas             = lags,
        trendOptions     = "rw/llt/srw/td",
        seasonalOptions  = "none/linear/equal",
        irregularOptions = "arma(0,0)"
    )
}

# .pts_call_uc: thin wrapper around UCompC() that dispatches on `command`.
# `args` is the list produced by .pts_uc_inputs (possibly with `p` swapped
# for a previously-estimated parameter vector for forecastOnly).
.pts_call_uc <- function(command, args){
    UCompC(command, args$y, args$u, args$model, args$h, args$lambda,
           args$outlier, args$tTest, args$criterion, args$periods, args$rhos,
           args$verbose, args$stepwise, args$p, args$arma, args$TVP,
           args$seas, args$trendOptions, args$seasonalOptions,
           args$irregularOptions,
           if (is.null(args$nsim)) 1L else as.integer(args$nsim),
           if (is.null(args$seed)) 0L else as.integer(args$seed))
}

# .pts_yindex: the time index of y, mirroring adam's `yIndex` construction
# at smooth/R/adam.R:519-531.  Returns a numeric / POSIXct / Date vector
# depending on the class of y.
.pts_yindex <- function(y){
    idx <- try(stats::time(y), silent = TRUE)
    if (!inherits(idx, "try-error") && length(idx) > 0) return(idx)
    seq_along(y)
}

# .pts_step: the inter-observation step inferred from yIndex; used to
# extrapolate forecast timestamps when y is zoo / Date / POSIXct.
.pts_step <- function(yIndex){
    if (length(yIndex) < 2) return(1)
    as.numeric(stats::median(diff(as.numeric(yIndex))))
}

# .pts_wrap_oos: wrap an out-of-sample numeric vector with the same time
# class as y.  Mirrors adam's pattern (smooth/R/adam.R:541-572):
#   * zoo input -> zoo(values, order.by = yIndex_oos)
#   * ts input  -> ts(values, start = yIndex[n] + step, frequency = freq(y))
#   * other     -> bare numeric
.pts_wrap_oos <- function(values, y){
    if (length(values) == 0) return(values)
    yClasses <- class(y)
    if (any(yClasses == "zoo")){
        yIndex <- .pts_yindex(y)
        step   <- .pts_step(yIndex)
        h      <- length(values)
        oosIdx <- utils::tail(yIndex, 1) + step * seq_len(h)
        return(zoo::zoo(as.numeric(values), order.by = oosIdx))
    }
    if (is.ts(y)){
        fy  <- stats::frequency(y)
        sy  <- stats::start(y, frequency = fy)
        aux <- stats::ts(rep(NA_real_, length(y) + 1L), start = sy, frequency = fy)
        return(stats::ts(as.numeric(values), start = stats::end(aux), frequency = fy))
    }
    as.numeric(values)
}

# Back-compat alias kept for forecast.pts and .pts_fit until they are
# updated; both now delegate to .pts_wrap_oos.
.pts_ts_forecast <- .pts_wrap_oos

# .pts_wrap_in: wrap an in-sample vector / matrix with the same time class
# as y.  Mirrors adam's pattern (smooth/R/adam.R:535-560).
.pts_wrap_in <- function(values, y, pad = 0L){
    if (length(values) == 0) return(values)
    yClasses <- class(y)
    if (any(yClasses == "zoo")){
        yIndex <- .pts_yindex(y)
        n      <- if (is.null(dim(values))) length(values) else nrow(values)
        start  <- length(y) - n + 1L + pad
        ord    <- yIndex[start:(start + n - 1L)]
        if (is.null(dim(values)))
            return(zoo::zoo(as.numeric(values), order.by = ord))
        return(zoo::zoo(unclass(values), order.by = ord))
    }
    if (is.ts(y)){
        fy <- stats::frequency(y)
        sy <- stats::start(y, frequency = fy)
        if (pad > 0L){
            aux <- stats::ts(rep(NA_real_, pad + 1L), start = sy, frequency = fy)
            return(stats::ts(values, start = stats::end(aux), frequency = fy))
        }
        return(stats::ts(values, start = sy, frequency = fy))
    }
    values
}

# .pts_ts_innov: wrap the innovations vector. Innovations are shorter than
# y by (length(y) - length(v)), so the wrapper starts (length(y) - length(v))
# steps later than y.
.pts_ts_innov <- function(values, y){
    if (length(values) == 0) return(values)
    pad <- length(y) - length(values)
    .pts_wrap_in(values, y, pad = pad)
}

# .pts_ts_comp: wrap the C++ component matrix (m x n stored column-major)
# as an (n+h) x m ts/zoo matrix anchored at start(y).
.pts_ts_comp <- function(raw, m, y, h){
    n <- length(raw) / m
    M <- t(matrix(raw, m, n))
    yClasses <- class(y)
    if (any(yClasses == "zoo")){
        yIndex <- .pts_yindex(y)
        step   <- .pts_step(yIndex)
        nObs   <- length(y)
        idx    <- c(yIndex[seq_len(min(n, nObs))],
                    utils::tail(yIndex, 1) + step * seq_len(max(0L, n - nObs)))
        return(zoo::zoo(M, order.by = idx))
    }
    if (is.ts(y)){
        fy <- stats::frequency(y)
        M  <- stats::ts(M, start = stats::start(y, frequency = fy), frequency = fy)
    }
    M
}

# .pts_wrap_states: wrap a state matrix (nrow = nobs + 1) anchoring row 1
# at the period BEFORE y starts (adam convention; smooth/R/adam.R:574).
.pts_wrap_states <- function(M, y){
    yClasses <- class(y)
    if (any(yClasses == "zoo")){
        yIndex <- .pts_yindex(y)
        step   <- .pts_step(yIndex)
        nObs   <- length(y)
        ord    <- c(yIndex[1L] - step, yIndex[seq_len(nObs)])
        return(zoo::zoo(unclass(M), order.by = ord))
    }
    if (is.ts(y)){
        fy <- stats::frequency(y)
        t0 <- stats::time(y)[1L] - 1 / fy
        return(stats::ts(unclass(M), start = t0, frequency = fy))
    }
    M
}

# .inv_box_cox: inverse Box-Cox transform.  Thresholds must match exactly
# what the C++ engine uses in BoxCox / invBoxCox (src/boxcox.h:34-58):
#   |lambda| < 0.02  -> exp(x)        (engine treats this as the log case)
#   lambda  > 0.98   -> x             (engine returns y unchanged)
#   otherwise        -> (lambda*x + 1)^(1/lambda)
# Preserves ts attributes on the input.
.inv_box_cox <- function(x, lambda){
    if (is.null(x) || length(x) == 0) return(x)
    if (lambda > 0.98) return(x)                       # identity, attrs intact
    out <- if (abs(lambda) < 0.02) exp(as.numeric(x))
           else                    (lambda * as.numeric(x) + 1) ^ (1 / lambda)
    if (inherits(x, "zoo"))
        out <- zoo::zoo(out, order.by = stats::time(x))
    else if (is.ts(x))
        out <- stats::ts(out, start = stats::start(x),
                         frequency = stats::frequency(x))
    out
}

# .pts_build_comp: take the engine's component matrix and rebuild it in
# the user-friendly column order  [Error, Fit, Level, Slope?, Seasonal?, ...].
# Fit becomes the row-sum of the structural components (so users can plot
# any subset and have them add up).
.pts_build_comp <- function(raw, v){
    nm  <- colnames(raw)
    ind <- c(1, which(nm == "Seasonal"), which(nm == "Slope"))
    pos <- max(ind) + as.integer(any(nm == "Irregular"))
    if (pos > length(nm))
        ind <- c(ind, (pos + 1L) : length(nm))
    comp <- cbind(v, raw[, 1], raw[, ind])
    comp[, 2] <- rowSums(comp[, 3 : ncol(comp), drop = FALSE])
    colnames(comp) <- c("Error", "Fit", nm[ind])
    comp
}

# .pts_fit: estimate the model and post-process the C++ result into the
# data shape that pts() uses for its returned object.  This replaces the
# (now retired) PTSsetup + MSOEsetup + MSOE chain.
.pts_fit <- function(y, u, model, lags, h, criterion, armaIdent, verbose){
    modelU <- pts_to_uc(model)
    # When the series has no seasonal frequency, fix lambda = 1 (no Box-Cox)
    # to match the behaviour the old PTSforecast() relied on.
    lambda <- modelU$lambda
    if (stats::frequency(y) == 1) lambda <- 1

    args <- .pts_uc_inputs(y = y, u = u, modelUC = modelU$modelU, h = h,
                           lambda = lambda, criterion = criterion, lags = lags,
                           verbose = verbose, armaIdent = armaIdent)
    out <- .pts_call_uc("all", args)
    if (identical(out$model, "error"))
        stop("Estimation failed in the C++ engine.", call. = FALSE)

    # Parameter vector and its covariance
    p    <- as.vector(out$coef)
    nPar <- length(p)
    names(p) <- out$parNames[seq_len(nPar)]
    covp <- out$covp
    if (!is.null(dim(covp))){
        rownames(covp) <- out$parNames[seq_len(nrow(covp))]
        colnames(covp) <- out$parNames[seq_len(ncol(covp))]
    }

    # Time-series wrappers.  yFor and yForV come out of the engine on the
    # Box-Cox scale; back-transform yFor for the user, keep yForV on the
    # BC scale so forecast.pts can compute intervals by endpoint transform.
    yFor_bc <- .pts_wrap_oos(out$yFor,  y)
    yForV   <- .pts_wrap_oos(out$yForV, y)
    yFor    <- .inv_box_cox(yFor_bc, out$lambda)
    v       <- .pts_ts_innov   (out$v,     y)

    # Component matrix (raw + user-friendly rearrangement).  Components stay
    # on the BC scale to keep the decomposition additive (Error + structural
    # cols = Fit); user-facing fitted / residuals get exposed separately.
    rawComp <- .pts_ts_comp(out$comp, out$m, y, h)
    colnames(rawComp) <- strsplit(out$compNames, "/")[[1]]
    comp <- .pts_build_comp(rawComp, v)
    # fitted on the original scale (back-transformed from comp[, "Fit"]).
    # residuals stay as the engine's BC-scale innovations -- they are the
    # white-noise sequence used by the validation table's diagnostics.
    # Both are truncated to in-sample length here (matching adam's storage
    # convention; see smooth/R/adam.R:558-560) so plot.smooth / cbind on
    # ts/zoo objects align naturally with actuals.
    ns        <- length(y)
    fittedBC  <- if (is.matrix(comp)) comp[, "Fit"]   else NA_real_
    residuals <- if (is.matrix(comp)) comp[, "Error"] else NA_real_
    fitted    <- if (is.matrix(comp)) .inv_box_cox(fittedBC, out$lambda) else NA_real_
    truncToNs <- function(x){
        if (length(x) <= ns) return(x)
        if (inherits(x, "zoo")) zoo::zoo(as.numeric(x)[seq_len(ns)],
                                         order.by = stats::time(x)[seq_len(ns)])
        else if (is.ts(x))      stats::window(x, end = stats::time(y)[ns])
        else                    x[seq_len(ns)]
    }
    fitted    <- truncToNs(fitted)
    residuals <- truncToNs(residuals)

    # MLE scale for the Gaussian (default) distribution.  Matches the
    # dnorm branch of smooth's scaler() at adam.R:1777:
    #   scale = sqrt( sum(errors^2) / obsInSample )
    # We use sum(na.rm=TRUE) so the leading filter-warmup NaNs are
    # ignored in the numerator, and length(y) as obsInSample.
    resVec  <- as.numeric(residuals)
    sumSq   <- sum(resVec[is.finite(resVec)] ^ 2)
    scale   <- if (length(y) > 0) sqrt(sumSq / length(y)) else NA_real_

    # Internal lags = the harmonic periods used inside the C++ engine; the
    # user will introduce a vector-valued `lags` argument later, at which
    # point lagsAll can mirror adam's per-parameter lag vector directly.
    lagsAll <- as.numeric(args$periods)

    # Information criteria
    crit <- as.numeric(out$criteria)
    if (length(crit) == 4L) names(crit) <- c("logLik", "AIC", "BIC", "AICc")

    # Stash the inputs UCompC will need for forecastOnly. y/u stay raw
    # (pre-BoxCox) because C++ re-applies BoxCox internally.
    forecast_args        <- args
    forecast_args$model  <- out$model     # the resolved (no '?') UC string
    forecast_args$lambda <- out$lambda
    forecast_args$p      <- p             # natural-scale variances + others
    forecast_args$periods<- out$periods
    forecast_args$rhos   <- out$rhos
    forecast_args$seas   <- lags

    list(
        modelUC      = out$model,
        lambda       = out$lambda,
        p            = p,
        p0           = as.vector(out$p0),
        covp         = covp,
        yFor         = yFor,           # original scale
        yForV        = yForV,          # BC scale (forecast.pts consumes it)
        v            = v,
        comp         = comp,           # BC scale, additive
        fitted       = fitted,         # original scale
        residuals    = residuals,      # BC scale (engine innovations)
        scale        = scale,          # MLE sigma on the BC scale
        lagsAll      = lagsAll,        # internal harmonic periods (C++ engine)
        table        = out$table,
        logLik       = if (length(crit) >= 1) unname(crit[1]) else NA_real_,
        IC           = if (length(crit) >= 4) crit[2:4]       else NA_real_,
        forecast_args = forecast_args
    )
}
