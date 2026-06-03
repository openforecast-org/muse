# Residual diagnostics for pts.  All read residuals(m), which lives on
# the Box-Cox scale of the engine's innovations -- by design (the residual
# autocorrelation / normality / outlier tests are only meaningful on the
# modelling scale, not after invBoxCox).

#' @export
rstandard.pts <- function(model, ...){
    e <- residuals(model)
    s <- sigma(model)
    if (s == 0 || !is.finite(s)) return(e * NA_real_)
    e / s
}

#' @export
rstudent.pts <- function(model, ...){
    # Leave-one-out studentisation for a state-space model is non-trivial
    # (would require recomputing sigma without each observation).  For
    # parity with smooth's rstudent.smooth -- which uses the same s for
    # all observations when no proper leverage is available -- we return
    # the standardised residuals as a conservative approximation.  smooth
    # itself does the equivalent inside plot.smooth's "rstudent" branch.
    rstandard.pts(model, ...)
}

#' @export
pointLik.pts <- function(object, log = TRUE, ...){
    e <- residuals(object)
    s <- sigma(object)
    if (s == 0 || !is.finite(s))
        return(e * NA_real_)
    dnorm(as.numeric(e), mean = 0, sd = s, log = log)
}

#' @export
outlierdummy.pts <- function(object,
                             level = 0.999,
                             type = c("rstandard", "rstudent"),
                             ...){
    type <- match.arg(type)
    r <- if (type == "rstandard") rstandard(object) else rstudent(object)
    q <- qnorm(level)
    ids <- which(abs(as.numeric(r)) > q)
    list(id        = ids,
         statistic = c(-q, q),
         type      = type,
         level     = level)
}
