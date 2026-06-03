# summary.pts -- richer, adam-style structured summary.  Returns a
# c("summary.pts", "list") with:
#   $coefficients   matrix: Estimate, Std. Error, t value, Pr(>|t|), Lower, Upper
#   $sigma, $logLik, $nobs, $nParam, $IC, $model, $modelUC, $lambda, $lags
#   $call, $timeElapsed
# Companion methods print.summary.pts and as.data.frame.summary.pts let
# users render it as text or pull the coefficient table for downstream use.

#' @export
summary.pts <- function(object, level = 0.95, ...){
    est <- coef(object)
    cv  <- vcov(object)
    if (is.null(dim(cv))){
        ses <- rep(NA_real_, length(est))
    } else {
        ses <- sqrt(diag(cv))
    }
    tval  <- est / ses
    pval  <- 2 * (1 - pnorm(abs(tval)))
    a     <- (1 - level) / 2
    z     <- qnorm(c(a, 1 - a))
    lower <- est + ses * z[1]
    upper <- est + ses * z[2]
    cmat  <- cbind(Estimate    = est,
                   `Std. Error` = ses,
                   `t value`    = tval,
                   `Pr(>|t|)`   = pval,
                   Lower        = lower,
                   Upper        = upper)
    rownames(cmat) <- names(est)
    out <- list(
        coefficients = cmat,
        sigma   = sigma(object),
        logLik  = as.numeric(logLik(object)),
        nobs    = nobs(object),
        nParam  = nparam(object),
        IC      = c(AIC  = AIC(object),
                    AICc = AICc(object),
                    BIC  = BIC(object),
                    BICc = BICc(object)),
        model   = object$model,
        modelUC = object$modelUC,
        lambda  = object$lambda,
        lags    = object$lags,
        level   = level,
        call    = object$call,
        timeElapsed = object$timeElapsed
    )
    class(out) <- c("summary.pts", "list")
    out
}

#' @export
print.summary.pts <- function(x, digits = 4, ...){
    cat("PTS state-space model\n")
    cat("  model:  ", x$model,
        "   (UC: ", x$modelUC, ")\n", sep = "")
    cat("  lambda: ", format(x$lambda, digits = digits),
        "   lags: ", x$lags,
        "   nobs: ", x$nobs,
        "   nParam: ", x$nParam, "\n", sep = "")
    cat("  sigma:  ", format(x$sigma, digits = digits), "\n", sep = "")
    cat("\nCoefficients (", format(100 * x$level, digits = digits), "% CI):\n", sep = "")
    cmat <- x$coefficients
    cmat_print <- cmat
    cmat_print[, "Estimate"]   <- signif(cmat[, "Estimate"], digits)
    cmat_print[, "Std. Error"] <- signif(cmat[, "Std. Error"], digits)
    cmat_print[, "t value"]    <- signif(cmat[, "t value"], digits)
    cmat_print[, "Pr(>|t|)"]   <- signif(cmat[, "Pr(>|t|)"], digits)
    cmat_print[, "Lower"]      <- signif(cmat[, "Lower"], digits)
    cmat_print[, "Upper"]      <- signif(cmat[, "Upper"], digits)
    print(as.data.frame(cmat_print), na.print = "NA")
    cat("\nlogLik:", format(x$logLik, digits = digits),
        "  AIC:",  format(x$IC["AIC"],  digits = digits),
        "  AICc:", format(x$IC["AICc"], digits = digits),
        "  BIC:",  format(x$IC["BIC"],  digits = digits),
        "  BICc:", format(x$IC["BICc"], digits = digits), "\n")
    invisible(x)
}

#' @export
as.data.frame.summary.pts <- function(x, row.names = NULL,
                                       optional = FALSE, ...){
    df <- as.data.frame(unclass(x$coefficients), row.names = row.names,
                         optional = optional, ...)
    df$Parameter <- rownames(x$coefficients)
    df[, c("Parameter", "Estimate", "Std. Error", "t value",
           "Pr(>|t|)", "Lower", "Upper")]
}
