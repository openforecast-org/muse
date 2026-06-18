# Internal helpers used by pts() and forecast.pts(). Not exported.

# .pts_guerrero_decomp_lambda: Box-Cox lambda screen via msdecompose +
# Guerrero CV minimisation.  Used by pts() when the user requests
# auto-lambda ("Z" in the model spec).  See the long comment in
# R/pts.R::pts where this is invoked for the full recipe; this is just
# the encapsulated implementation.
#
# Returns a numeric lambda in [lambda_lower, lambda_upper].  Falls back
# to 1 (identity) on any precondition failure.
.pts_guerrero_decomp_lambda <- function(y, lags,
                                        lambda_lower = 0,
                                        lambda_upper = 2){
    yv <- as.numeric(y)
    if (any(!is.finite(yv)) || any(yv <= 0)) return(1)
    m <- as.integer(utils::tail(as.integer(lags), 1L))
    if (length(m) == 0L || is.na(m) || m < 2L) return(1)
    n <- length(yv)
    if (n < 2L * m) return(1)

    decomp <- tryCatch(
        smooth::msdecompose(yv, lags = m, type = "additive",
                            smoother = "ma"),
        error = function(e) NULL)
    if (is.null(decomp)) return(1)

    mu_t <- as.numeric(decomp$states[, 1L])
    R    <- n %/% m
    idx  <- rep(seq_len(R), each = m)
    keep <- seq_len(R * m)
    mu_b <- as.numeric(tapply(mu_t[keep],            idx, mean, na.rm = TRUE))
    sd_b <- as.numeric(tapply((yv - mu_t)[keep],     idx, sd,   na.rm = TRUE))
    ok   <- is.finite(mu_b) & is.finite(sd_b) & mu_b > 0 & sd_b > 0
    mu_b <- mu_b[ok]; sd_b <- sd_b[ok]
    if (length(mu_b) < 2L) return(1)

    if (lambda_lower >= lambda_upper) return(lambda_lower)
    obj <- function(L){
        r <- sd_b * mu_b^(L - 1)
        if (any(!is.finite(r))) return(Inf)
        sd(r) / mean(r)
    }
    opt <- tryCatch(stats::optimize(obj, lower = lambda_lower,
                                    upper = lambda_upper, tol = 1e-4),
                    error = function(e) NULL)
    if (is.null(opt) || !is.finite(opt$objective)) return(1)
    opt$minimum
}


# .pts_parse_data: extract a response vector + regressor matrix from any of
# the input shapes pts() accepts.  Returns
#   list(y = <response>, u = <NULL or k x n matrix>,
#        responseName = <string>, formula = <formula or NULL>)
#
# Behaviour matches adam-style data handling (smooth/R/adamGeneral.R:56-119)
# but PTS supports only the univariate-response case: matrix / data.frame
# inputs use column 1 as the response and the remaining columns as xreg
# unless a `formula` is supplied to pin them explicitly.
.pts_parse_data <- function(data, formula = NULL){
    # Vector / ts / zoo -> univariate response, no xreg
    if (is.null(dim(data))){
        if (!is.null(formula))
            stop("formula is only meaningful when `data` is a matrix or ",
                 "data.frame.", call. = FALSE)
        return(list(y = data, u = NULL, responseName = "y", formula = NULL))
    }

    if (!is.null(formula)){
        # Build a model frame so categorical xregs / interactions just work
        mf <- stats::model.frame(formula, data = as.data.frame(data))
        responseName <- all.vars(formula)[1L]
        y    <- stats::model.response(mf)
        mm   <- stats::model.matrix(stats::terms(mf), mf)
        keep <- !colnames(mm) %in% "(Intercept)"
        u    <- if (any(keep)) t(mm[, keep, drop = FALSE]) else NULL
    } else {
        # No formula: column 1 is response, columns 2..k are xreg
        if (is.data.frame(data)) data <- as.matrix(data)
        responseName <- if (!is.null(colnames(data))) colnames(data)[1L]
                        else "y"
        y <- data[, 1L]
        u <- if (ncol(data) > 1L) t(data[, -1L, drop = FALSE]) else NULL
    }
    list(y = y, u = u, responseName = responseName, formula = formula)
}

# .pts_uc_inputs: marshal y / u / model / horizon / options into the exact
# argument list the C++ .UCompC entry point expects. Returns a named list
# that we pass to .UCompC and also stash on the pts object for forecast.pts.
.pts_uc_inputs <- function(y, u, modelUC, h, lambda, criterion, lags,
                           verbose, armaIdent,
                           irregularOptions = "arma(0,0)",
                           outlier = 0,
                           lambdaLower = -Inf){
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
        outlier          = as.numeric(outlier),
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
        irregularOptions = irregularOptions,
        lambdaLower      = as.numeric(lambdaLower)
    )
}

# .pts_call_uc: thin wrapper around .UCompC() that dispatches on `command`.
# `args` is the list produced by .pts_uc_inputs (possibly with `p` swapped
# for a previously-estimated parameter vector for forecastOnly).
.pts_call_uc <- function(command, args){
    .UCompC(command, args$y, args$u, args$model, args$h, args$lambda,
           args$outlier, args$tTest, args$criterion, args$periods, args$rhos,
           args$verbose, args$stepwise, args$p, args$arma, args$TVP,
           args$seas, args$trendOptions, args$seasonalOptions,
           args$irregularOptions,
           if (is.null(args$nsim)) 1L else as.integer(args$nsim),
           if (is.null(args$seed)) 0L else as.integer(args$seed),
           if (is.null(args$lambdaLower)) -Inf else as.numeric(args$lambdaLower))
}

# .pts_resolve_class: mirror adam's input-class resolution at
# smooth/R/adamGeneral.R:32-168.  Returns a list with:
#   yIndex     - the time index of y (numeric / Date / POSIXct vector,
#                length == length(y)), with try(time(y)) + fallbacks.
#   yClasses   - the resolved input-class vector; bare numeric / integer
#                gets promoted to "ts" (default) or "zoo" if yIndex is
#                Date / POSIXct (matches adamGeneral.R:160-168).
#   yFrequency - frequency(y), used by the ts wrappers.
#   step       - inter-observation step, used to extrapolate OOS indices
#                for zoo / Date / POSIXct inputs.
.pts_resolve_class <- function(y){
    yIndex <- try(stats::time(y), silent = TRUE)
    if (inherits(yIndex, "try-error") || length(yIndex) == 0){
        yIndex <- seq_along(y)
    }
    yClasses <- class(y)
    if (all(yClasses %in% c("integer", "numeric", "matrix"))){
        if (any(class(yIndex) %in% c("POSIXct", "Date"))){
            yClasses <- "zoo"
        } else {
            yClasses <- "ts"
        }
    }
    yFrequency <- stats::frequency(y)
    step <- if (length(yIndex) >= 2)
                as.numeric(stats::median(diff(as.numeric(yIndex))))
            else 1
    list(yIndex = yIndex, yClasses = yClasses,
         yFrequency = yFrequency, step = step)
}

# .pts_yindex / .pts_step are thin back-compat wrappers around the
# resolver, kept so .pts_call_uc users don't have to thread the whole
# resolved list through every call site.
.pts_yindex <- function(y) .pts_resolve_class(y)$yIndex
.pts_step   <- function(yIndex){
    if (length(yIndex) < 2) return(1)
    as.numeric(stats::median(diff(as.numeric(yIndex))))
}

# .pts_wrap_oos: wrap an out-of-sample vector.  Adam's pattern (the line
# numbers below cite smooth/R/adam.R):
#   * yClasses contains "ts"  -> ts(values, start = yIndex[n] + step,
#                                  frequency = yFrequency)  (line 567)
#   * otherwise               -> zoo(values, order.by = oosIdx)  (line 545)
.pts_wrap_oos <- function(values, y){
    if (length(values) == 0) return(values)
    r <- .pts_resolve_class(y)
    if (any(r$yClasses == "ts")){
        startOos <- utils::tail(as.numeric(r$yIndex), 1) + r$step
        return(stats::ts(as.numeric(values), start = startOos,
                         frequency = r$yFrequency))
    }
    oosIdx <- utils::tail(r$yIndex, 1) + r$step * seq_along(values)
    zoo::zoo(as.numeric(values), order.by = oosIdx)
}

# .pts_wrap_in: wrap an in-sample vector / matrix.  Adam's pattern
# (smooth/R/adam.R):
#   * yClasses contains "ts" -> ts(values, start = yStart + pad/freq,
#                                  frequency = yFrequency)        (line 559)
#   * otherwise              -> zoo(values, order.by = yIndex slice) (line 536)
# `pad` skips that many leading observations of y (used by the innovations
# wrapper, which shifts later than y when the engine drops warm-up rows).
.pts_wrap_in <- function(values, y, pad = 0L){
    if (length(values) == 0) return(values)
    r <- .pts_resolve_class(y)
    if (any(r$yClasses == "ts")){
        sy <- if (is.ts(y)) stats::start(y, frequency = r$yFrequency)
              else          as.numeric(r$yIndex)[1L]
        if (pad > 0L){
            aux <- stats::ts(rep(NA_real_, pad + 1L), start = sy,
                             frequency = r$yFrequency)
            return(stats::ts(values, start = stats::end(aux),
                             frequency = r$yFrequency))
        }
        return(stats::ts(values, start = sy, frequency = r$yFrequency))
    }
    n     <- if (is.null(dim(values))) length(values) else nrow(values)
    start <- pad + 1L
    ord   <- r$yIndex[start:(start + n - 1L)]
    if (is.null(dim(values)))
        return(zoo::zoo(as.numeric(values), order.by = ord))
    zoo::zoo(unclass(values), order.by = ord)
}

# .pts_ts_innov: wrap the innovations vector.  Two engine paths feed in:
#   * v shorter than y by k -- the KF dropped k warm-up rows.  Shift the
#     time index forward by k (positive pad).
#   * v of length n + h     -- the engine appended forecast-tail
#     placeholders.  Residuals are an in-sample concept, so drop the
#     last h entries before wrapping; the resulting pad is zero.
# `[.Date` / `[.yearmon` error on negative subscripts (rather than
# padding with NA the way base `[` does), so we never let pad be
# negative when entering .pts_wrap_in.
.pts_ts_innov <- function(values, y){
    if (length(values) == 0) return(values)
    ny <- length(y)
    if (length(values) > ny) values <- values[seq_len(ny)]
    pad <- ny - length(values)
    .pts_wrap_in(values, y, pad = pad)
}

# .pts_ts_comp: wrap the C++ component matrix (m x n stored column-major)
# as an (n+h) x m ts/zoo matrix anchored at start(y).  Same dispatch as
# above; OOS rows get extrapolated indices for the zoo branch.
.pts_ts_comp <- function(raw, m, y, h){
    n <- length(raw) / m
    M <- t(matrix(raw, m, n))
    r <- .pts_resolve_class(y)
    if (any(r$yClasses == "ts")){
        sy <- if (is.ts(y)) stats::start(y, frequency = r$yFrequency)
              else          as.numeric(r$yIndex)[1L]
        return(stats::ts(M, start = sy, frequency = r$yFrequency))
    }
    nObs <- length(y)
    idx  <- c(r$yIndex[seq_len(min(n, nObs))],
              utils::tail(r$yIndex, 1) + r$step * seq_len(max(0L, n - nObs)))
    zoo::zoo(M, order.by = idx)
}

# .pts_wrap_states: wrap a state matrix (nrow = nobs + 1) anchoring row 1
# at the period BEFORE y starts.  Adam's convention at adam.R:574 (ts
# branch) and adam.R:554 (zoo branch).
.pts_wrap_states <- function(M, y){
    r <- .pts_resolve_class(y)
    if (any(r$yClasses == "ts")){
        t0 <- if (is.ts(y)) stats::time(y)[1L] - 1 / r$yFrequency
              else          as.numeric(r$yIndex)[1L] - r$step
        return(stats::ts(unclass(M), start = t0, frequency = r$yFrequency))
    }
    nObs <- length(y)
    ord  <- c(r$yIndex[1L] - r$step, r$yIndex[seq_len(nObs)])
    zoo::zoo(unclass(M), order.by = ord)
}

# .inv_box_cox: inverse Box-Cox transform.  Branches must match the
# C++ engine in BoxCox / invBoxCox (src/boxcox.h) and bcnorm.h exactly:
# both sides use exact-equality switches at the two singular points
# (lambda == 1 -> identity; lambda == 0 -> exp), not thresholds.
#   lambda == 0  -> exp(x)
#   lambda == 1  -> x
#   otherwise    -> (lambda*x + 1)^(1/lambda)
# Preserves ts attributes on the input.
.inv_box_cox <- function(x, lambda){
    if (is.null(x) || length(x) == 0) return(x)
    if (lambda == 1) return(x)                         # identity, attrs intact
    xv <- as.numeric(x)
    if (lambda == 0){
        out <- exp(xv)                                 # log case; -Inf -> 0, +Inf -> +Inf
    } else {
        # Box-Cox support: Y on the original scale requires
        # 1 + lambda*X > 0.  When `xv` is at or below this boundary
        # (e.g. -Inf from a one-sided "upper" interval), the original
        # scale value is at the support boundary: 0 for lambda > 0
        # (Y >= 0), or +Inf for lambda < 0 (Y unbounded below in BC).
        arg <- lambda * xv + 1
        out <- ifelse(arg > 0, arg ^ (1 / lambda),
                      ifelse(lambda > 0, 0, Inf))
    }
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
    ind <- c(1, which(nm == "Slope"), which(nm == "Seasonal"))
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
.pts_fit <- function(y, u, model, lags, h, criterion, armaIdent, verbose,
                     ar = 0L, ma = 0L, armaLags = 1L, outlier = 0,
                     lambdaLower = -Inf, B = NULL){
    # Flatten per-lag ar / ma vectors into the format pts_to_uc consumes:
    #   length-1 lags → c(p, q)         (non-seasonal arma(p,q))
    #   length-2 lags → c(p, q, P, Q, s) (sarma(p,q)(P,Q)_s)
    armaOrders <- if (length(armaLags) == 1L) c(ar[1L], ma[1L])
                  else c(ar[1L], ma[1L], ar[2L], ma[2L], armaLags[2L])
    modelU <- pts_to_uc(model, armaOrders = armaOrders,
                        armaSelect = armaIdent)
    lambda <- modelU$lambda

    args <- .pts_uc_inputs(y = y, u = u, modelUC = modelU$modelU, h = h,
                           lambda = lambda, criterion = criterion, lags = lags,
                           verbose = verbose, armaIdent = armaIdent,
                           irregularOptions =
                               .pts_arma_candidates(ar, ma, armaLags,
                                                    armaIdent),
                           outlier = outlier,
                           lambdaLower = lambdaLower)
    # Override the default sentinel (-9999.9) when the caller supplied an
    # explicit starting vector via `B` (adam-style internal hatch, passed
    # through pts(... , B = ...) and used by the loss-surface experiment
    # for multi-start optimisation).  Natural-scale (positive variances);
    # the engine's userP0 branch in initParBsm converts to log-ratio.
    if (!is.null(B) && length(B) > 0)
        args$p <- as.numeric(B)
    out <- .pts_call_uc("all", args)
    if (identical(out$model, "error"))
        stop("Estimation failed in the C++ engine.", call. = FALSE)
    # lambdaEstimated is decided inside the engine: profile-lambda's snap
    # test sets it to FALSE when lambda landed on an anchor (no +1 DoF) and
    # TRUE when the optimised lambda* won (+1 DoF).  Fixed-lambda specs
    # (e.g. "1NT", "0NT") always come back FALSE.
    lambdaEstimated <- isTRUE(out$lambdaEstimated)

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
    # dnorm branch of smooth's scaler() at adam.R:1777 verbatim:
    #   scale = sqrt( sum(errors^2) / obsInSample )
    # obsInSample is the in-sample (training) length; leading filter-warmup
    # NaNs are dropped from the numerator (sum() is na.rm-equivalent on the
    # already-filtered finite vector).
    obsInSample <- length(y)
    resVec      <- as.numeric(residuals)
    sumSq       <- sum(resVec[is.finite(resVec)] ^ 2)
    scale       <- if (obsInSample > 0) sqrt(sumSq / obsInSample) else NA_real_

    # Internal lags = the harmonic periods used inside the C++ engine; the
    # user will introduce a vector-valued `lags` argument later, at which
    # point lagsAll can mirror adam's per-parameter lag vector directly.
    lagsAll <- as.numeric(args$periods)

    # Information criteria.  The engine computes the Box-Cox-corrected
    # log-likelihood directly (PTSmodel.h:1130 calls bcnormLogDensity on
    # the un-transformed response), so crit[0] is already on the original
    # response scale -- we just propagate it.
    crit   <- as.numeric(out$criteria)
    logLik <- if (length(crit) >= 1) unname(crit[1]) else NA_real_
    if (length(crit) == 4L) names(crit) <- c("logLik", "AIC", "BIC", "AICc")

    # Outlier detection — engine emits a (nDetected x 2) matrix with
    # columns (type, time) where type ∈ {0 = AO, 1 = LS, 2 = SC} and time
    # is the 0-based index in y.  Convert to a user-friendly data frame.
    detected <- out$typeOutliers
    outliersDetected <- if (!is.null(detected) && length(detected) > 0L &&
                            nrow(detected) > 0L){
        types <- c("AO", "LS", "SC")[as.integer(detected[, 1L]) + 1L]
        # Times come back zero-indexed from the engine; flip to 1-based
        # to match R conventions / the user's series positions.
        data.frame(time = as.integer(detected[, 2L]) + 1L,
                   type = factor(types, levels = c("AO", "LS", "SC")),
                   stringsAsFactors = FALSE)
    } else {
        data.frame(time = integer(0),
                   type = factor(character(0),
                                 levels = c("AO", "LS", "SC")),
                   stringsAsFactors = FALSE)
    }

    list(
        modelUC      = out$model,
        lambda       = out$lambda,
        p            = p,
        p0           = as.vector(out$p0),
        covp         = covp,
        yFor         = yFor,           # original scale (length h, possibly 0)
        v            = v,
        comp         = comp,           # BC scale, additive
        fitted       = fitted,         # original scale
        residuals    = residuals,      # BC scale (engine innovations)
        scale        = scale,          # MLE sigma on the BC scale
        lagsAll      = lagsAll,        # internal harmonic periods (C++ engine)
        table        = out$table,
        logLik       = logLik,
        lambdaEstimated = lambdaEstimated,
        IC           = if (length(crit) >= 4) crit[2:4]       else NA_real_,
        outliersDetected = outliersDetected
    )
}

# .pts_forecast_inputs: rebuild the .UCompC argument list from a fitted
# pts object, using slot values directly so we don't need a separate
# $forecast_args cache.  Used by forecast.pts (forecastOnly path).
#
# When the model has regressors (object$u is non-NULL), forecast.pts
# requires `newdata` with at least `h` rows of future xreg values; this
# helper concatenates object$u (k x n) with t(newdata)[, 1:h] (k x h)
# into the (k x (n + h)) matrix the engine wants -- the C++ side reads
# `u.col(n + i)` at SSpace.h:286 during the forecast loop.
.pts_forecast_inputs <- function(object, h, newdata = NULL){
    u <- object$u
    if (is.null(u)){
        if (!is.null(newdata))
            stop("`newdata` was supplied but the fitted model has no ",
                 "regressors.", call. = FALSE)
        u_mat <- matrix(0, 1, 2)
    } else {
        # u is k x n_train (built that way in .pts_parse_data)
        if (is.null(newdata))
            stop("This model was fitted with regressors; please supply ",
                 "`newdata` containing at least `h` rows of future ",
                 "regressor values.", call. = FALSE)
        # Bring newdata into the same (rows = obs, cols = vars) layout as
        # the original `data` matrix the user passed to pts(); then
        # transpose to k x h for column-bind with the engine's `u`.
        if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
        if (is.null(dim(newdata)))  newdata <- matrix(newdata, ncol = 1L)
        if (nrow(newdata) < h)
            stop("`newdata` must have at least `h` rows of regressor ",
                 "values; got ", nrow(newdata), " for h = ", h, ".",
                 call. = FALSE)
        if (ncol(newdata) != nrow(u))
            stop("`newdata` has ", ncol(newdata), " column(s); the fitted ",
                 "model expects ", nrow(u), " (one column per regressor).",
                 call. = FALSE)
        u_future <- t(newdata[seq_len(h), , drop = FALSE])
        u_mat    <- cbind(u, u_future)
    }
    list(
        y                = as.numeric(object$data),
        u                = u_mat,
        model            = object$modelUC,
        h                = as.integer(h),
        lambda           = object$lambda,
        outlier          = 0,
        tTest            = FALSE,
        criterion        = "aic",
        periods          = as.numeric(object$lagsAll),
        rhos             = rep(1, length(object$lagsAll)),
        verbose          = FALSE,
        stepwise         = FALSE,
        p                = as.numeric(object$B),
        arma             = FALSE,
        TVP              = -9999.99,
        seas             = object$lags,
        trendOptions     = "rw/llt/srw/td",
        seasonalOptions  = "none/linear/equal",
        irregularOptions = .pts_arma_candidates(
            if (is.null(object$orders$ar))   0L else object$orders$ar,
            if (is.null(object$orders$ma))   0L else object$orders$ma,
            if (is.null(object$orders$lags)) 1L else object$orders$lags,
            FALSE)
    )
}
