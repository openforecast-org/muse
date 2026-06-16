# confint.pts -- standard Wald confidence intervals from vcov(object).
# Parameters whose vcov is non-finite (e.g. the concentrated-out variance
# whose Hessian entry is undefined) get NA bounds with a warning.

#' @export
confint.pts <- function(object, parm, level = 0.95, ...){
    est  <- coef(object)
    cv   <- vcov(object)
    if (is.null(dim(cv))){
        ses <- rep(NA_real_, length(est))
    } else {
        ses <- sqrt(diag(cv))
    }
    if (missing(parm)){
        parm <- seq_along(est)
    } else if (is.character(parm)){
        parm <- match(parm, names(est))
    }
    a <- (1 - level) / 2
    z <- qnorm(c(a, 1 - a))
    out <- cbind(est[parm] + ses[parm] * z[1],
                 est[parm] + ses[parm] * z[2])
    colnames(out) <- sprintf("%.1f %%", 100 * c(a, 1 - a))
    rownames(out) <- names(est)[parm]
    nNa <- sum(!is.finite(rowSums(out)))
    if (nNa > 0)
        warning(sprintf(
            "%d parameter(s) have non-finite vcov (e.g. concentrated-out variance); CI returned as NA.",
            nNa), call. = FALSE)
    out
}
