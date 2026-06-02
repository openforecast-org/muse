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
           args$irregularOptions)
}

# .pts_ts_forecast: wrap a forecast vector as a ts object whose first
# observation lies one period after the last observation of y.
.pts_ts_forecast <- function(values, y){
    if (!is.ts(y) || length(values) == 0) return(values)
    fy  <- stats::frequency(y)
    sy  <- stats::start(y, frequency = fy)
    aux <- stats::ts(rep(NA_real_, length(y) + 1L), start = sy, frequency = fy)
    stats::ts(values, start = stats::end(aux), frequency = fy)
}

# .pts_ts_innov: wrap the innovations vector. Innovations are shorter than
# y by (length(y) - length(v)), so the ts starts later than y.
.pts_ts_innov <- function(values, y){
    if (!is.ts(y) || length(values) == 0) return(values)
    fy  <- stats::frequency(y)
    sy  <- stats::start(y, frequency = fy)
    pad <- length(y) - length(values) + 1L
    aux <- stats::ts(rep(NA_real_, pad), start = sy, frequency = fy)
    stats::ts(values, start = stats::end(aux), frequency = fy)
}

# .pts_ts_comp: wrap the C++ component matrix (m x n stored column-major)
# as an (n+h) x m ts matrix anchored at start(y).
.pts_ts_comp <- function(raw, m, y, h){
    n <- length(raw) / m
    M <- t(matrix(raw, m, n))
    if (is.ts(y))
        M <- stats::ts(M, start = stats::start(y, frequency = stats::frequency(y)),
                       frequency = stats::frequency(y))
    M
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

    # Time-series wrappers
    yFor  <- .pts_ts_forecast(out$yFor,  y)
    yForV <- .pts_ts_forecast(out$yForV, y)
    v     <- .pts_ts_innov   (out$v,     y)

    # Component matrix (raw + user-friendly rearrangement)
    rawComp <- .pts_ts_comp(out$comp, out$m, y, h)
    colnames(rawComp) <- strsplit(out$compNames, "/")[[1]]
    comp <- .pts_build_comp(rawComp, v)

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
        yFor         = yFor,
        yForV        = yForV,
        v            = v,
        comp         = comp,
        table        = out$table,
        logLik       = if (length(crit) >= 1) unname(crit[1]) else NA_real_,
        IC           = if (length(crit) >= 4) crit[2:4]       else NA_real_,
        forecast_args = forecast_args
    )
}
