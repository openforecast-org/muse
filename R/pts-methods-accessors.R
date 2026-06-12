# S3 method accessors for pts that mirror the smooth / greybox / stats
# generics adam exposes.  All one-liners reading existing slots.  The
# residual-related ones (sigma, extractSigma) read residuals(m), which
# is the Box-Cox-scale innovation vector -- intentional, matches what
# the validation table's diagnostics already assume.

# sigma.pts -- mirrors sigma.adam (smooth/R/adam.R:6398-6431) for dnorm:
#   sigma = sqrt( sum(residuals^2, na.rm=TRUE) / (obsInSample - nparam) )
# The denominator is obsInSample (the in-sample / training length, i.e.
# nobs(object, all = FALSE)), NOT obsAll.  Uses df = obsInSample - k and
# does not subtract mean(residuals).
#' @export
sigma.pts <- function(object, ...){
    obsInSample <- nobs(object, all = FALSE)
    df          <- obsInSample - nparam(object)
    if (df <= 0) df <- obsInSample
    sqrt(sum(residuals(object) ^ 2, na.rm = TRUE) / df)
}

#' @export
extractSigma.pts <- function(object, ...) sigma.pts(object)

#' @export
extractScale.pts <- function(object, ...) object$scale

#' @export
nparam.pts <- function(object, ...) object$nParam

#' @export
actuals.pts <- function(object, all = TRUE, ...) object$data

#' @export
actuals.pts.forecast <- function(object, all = TRUE, ...) actuals.pts(object$model)

#' @export
modelType.pts <- function(object, ...) object$model

#' @export
modelName.pts <- function(object, ...){
    # Translate the PTS(lambda,T,S) code into a human-readable label, e.g.
    # PTS(0,N,T) -> "PTS(Power=0, Trend=None, Seasonal=Trigonometric)"
    m <- object$model
    inside <- sub("^PTS\\((.*)\\)$", "\\1", m)
    parts  <- strsplit(inside, ",", fixed = TRUE)[[1]]
    if (length(parts) != 3) return(m)
    trendMap <- c(N = "None", L = "Local-linear", D = "Damped-trend",
                  G = "Global-trend", Z = "Auto")
    seasMap  <- c(N = "None", T = "Trigonometric", D = "Discrete", Z = "Auto")
    sprintf("PTS(Power=%s, Trend=%s, Seasonal=%s)",
            parts[1],
            trendMap[parts[2]],
            seasMap[parts[3]])
}

#' @export
lags.pts <- function(object, ...) object$lags

#' @export
orders.pts <- function(object, ...){
    # Pull the ARMA(p,q) of the irregular component out of modelUC.
    m <- object$modelUC
    p1 <- regexpr("arma", m, fixed = TRUE)[1]
    if (p1 == -1) return(list(ar = 0, i = 0, ma = 0))
    ar <- suppressWarnings(as.integer(
        sub("^arma\\(([0-9]+),([0-9]+)\\)$", "\\1",
            substring(m, p1))))
    ma <- suppressWarnings(as.integer(
        sub("^arma\\(([0-9]+),([0-9]+)\\)$", "\\2",
            substring(m, p1))))
    if (is.na(ar)) ar <- 0L
    if (is.na(ma)) ma <- 0L
    list(ar = ar, i = 0L, ma = ma)
}

#' Initial state values for a fitted \code{pts} object.
#'
#' Returns the smoothed structural states at the first observation, with
#' the \code{"Error"} and \code{"Fit"} columns dropped.  For deterministic
#' components -- e.g. the slope under the global (\code{G}) trend, where
#' the slope is fixed throughout the horizon -- this equals the t = 0
#' initial value; for stochastic components it is the smoother's
#' estimate at t = 1, a close proxy to the t = 0 initial.
#'
#' @param object A fitted object of class \code{"pts"}.
#' @param ... Unused.
#' @return Named numeric vector of initial structural-state values.
#' @export
initials <- function(object, ...) UseMethod("initials")

#' @rdname pts-methods
#' @export
initials.pts <- function(object, ...){
    if (is.null(object$comp) || !is.matrix(object$comp))
        return(numeric(0))
    cols <- setdiff(colnames(object$comp), c("Error", "Fit"))
    if (length(cols) == 0) return(numeric(0))
    vals <- as.numeric(object$comp[1, cols])
    names(vals) <- cols
    vals
}

#' @export
errorType.pts <- function(object, ...){
    # The state-space model fits additive innovations on the Box-Cox scale.
    # On the original scale this corresponds to a Box-Cox additive error,
    # but smooth's "A" / "M" dichotomy maps cleanest to "A" here -- the
    # transform is metadata captured by object$lambda.
    "A"
}

# AIC / BIC / AICc / BICc methods intentionally NOT defined for class
# `pts`.  Dispatch falls through the c("pts", "smooth") class chain so:
#   AIC(m)  ->  stats::AIC.default   (via logLik.pts attributes)
#   BIC(m)  ->  stats::BIC.default   (via logLik.pts attributes)
#   AICc(m) ->  greybox / smooth::AICc.smooth
#   BICc(m) ->  greybox / smooth::BICc.smooth
# All four use logLik(object), nparam(object), and nobs(object) (default
# all = FALSE).  Keeping pts as a thin user of the existing greybox /
# stats formulas avoids drifting from the smooth ecosystem.
