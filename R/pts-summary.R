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
    isIrr <- names(est) == "Irregular"
    cmat  <- cbind(Estimate    = est[!isIrr],
                   `Std. Error` = ses[!isIrr],
                   `t value`    = tval[!isIrr],
                   `Pr(>|t|)`   = pval[!isIrr],
                   Lower        = lower[!isIrr],
                   Upper        = upper[!isIrr])
    rownames(cmat) <- names(est)[!isIrr]

    # Variance proportions + delta-method SEs.
    # Only structural variance params (not Damping, not Irregular, not ARMA, not Beta).
    nm <- names(est)
    isArma  <- grepl("^(AR|MA)\\(", nm)
    isXreg  <- grepl("^Beta",       nm)
    isDamp  <- nm == "Damping"
    isIrr   <- nm == "Irregular"
    isVar   <- !(isArma | isXreg | isDamp | isIrr)
    varVals <- est[isVar]
    S       <- sum(varVals)
    props   <- if (length(varVals) > 0 && S > 0) varVals / S else varVals

    propSEs <- rep(NA_real_, length(varVals))
    names(propSEs) <- names(varVals)
    if (length(varVals) > 1 && !is.null(dim(cv)) && S > 0){
        varIdx <- which(isVar)
        if (max(varIdx) <= nrow(cv)){
            Sv <- cv[varIdx, varIdx, drop = FALSE]
            # Jacobian J[i,j] = dp_i/dv_j (delta method)
            J  <- (diag(length(varVals)) - outer(rep(1, length(varVals)), props)) / S
            propVar <- diag(J %*% Sv %*% t(J))
            propSEs <- sqrt(pmax(0, propVar))
        }
    }
    propMat <- cbind(Proportion  = props,
                     `Std. Error` = propSEs)
    rownames(propMat) <- names(props)

    out <- list(
        coefficients = cmat,
        proportions  = propMat,
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
        lagsAll = object$lagsAll,
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
    perStr <- if (!is.null(x$lagsAll) && length(x$lagsAll) > 1)
                  paste(formatC(x$lagsAll, format = "f", digits = 1), collapse = " / ")
              else
                  as.character(x$lags)
    cat("  lambda: ", format(x$lambda, digits = digits),
        "   periods: ", perStr,
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

    if (!is.null(x$proportions) && nrow(x$proportions) > 0){
        cat("\nVariance proportions:\n")
        pmat_print <- x$proportions
        pmat_print[, "Proportion"]  <- signif(x$proportions[, "Proportion"],  digits)
        pmat_print[, "Std. Error"]  <- signif(x$proportions[, "Std. Error"],  digits)
        print(as.data.frame(pmat_print), na.print = "NA")
    }

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
