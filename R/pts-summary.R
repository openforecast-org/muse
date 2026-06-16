# summary.pts -- adam-style structured summary.  Returns a
# c("summary.pts", "list") with:
#   $coefficients   matrix: Estimate, Std. Error, Lower, Upper
#   $proportions    matrix: Proportion, Std. Error
#   $concentrated   names of variance rows held fixed at the FOC
#   $sigma, $logLik, $nobs, $nParam, $IC, $model, $modelUC, $lambda, $lags
#   $responseName, $isTrig, $loss, $lossValue, $call, $timeElapsed
# Companion methods print.summary.pts and as.data.frame.summary.pts let
# users render it as text or pull the coefficient table for downstream use.

# Helper: when the trend is the global / deterministic one (UC code "td",
# spelled "G" in the PTS letter form), the engine does not push a "Slope"
# entry to the parameter vector because there is no slope shock to estimate
# the variance of -- but the slope itself is identified by the augmented
# KF and is constant across the horizon (initials(object)["Slope"]).  We
# inject a row carrying that value into summary / print tables so the user
# can read the drift on the same canvas as the other parameters.
.pts_det_slope <- function(object){
    if (is.null(object$modelUC) || !startsWith(object$modelUC, "td/"))
        return(NULL)
    if (is.null(object$comp) || !is.matrix(object$comp) ||
        !"Slope" %in% colnames(object$comp))
        return(NULL)
    as.numeric(object$comp[1, "Slope"])
}

# Pull the response name from the formula when available; otherwise look
# at the call (positional or named `data` arg).  Mirrors adam's
# all.vars(formula(object))[1] but works when pts is called with a bare
# time series.
.pts_response_name <- function(object){
    if (!is.null(object$formula))
        return(all.vars(object$formula)[1])
    if (!is.null(object$responseName) && nzchar(object$responseName) &&
        object$responseName != "y")
        return(object$responseName)
    cl <- object$call
    if (is.null(cl)) return("y")
    if (!is.null(cl$data))             return(deparse(cl$data))
    if (length(cl) >= 2)               return(deparse(cl[[2]]))
    "y"
}

#' @export
summary.pts <- function(object, level = 0.95, ...){
    est <- coef(object)
    cv  <- vcov(object)
    n   <- nobs(object)
    nm  <- names(est)

    # Concentrated-out variance(s): the optimiser holds these fixed at the
    # FOC value, so the inverse Hessian has NaN on the diagonal.  The
    # analytical SE for the MLE of a Gaussian variance is sigma^2 * sqrt(2/n)
    # -- substitute it in so the row is informative instead of all-NaN, and
    # patch the joint vcov so the delta method on proportions works (we
    # treat the concentrated variance as independent of the others at the
    # asymptotic level; the joint Hessian does not expose the cross terms).
    #
    # `cv` may carry FEWER rows than `est`: under outliers = "use" the
    # engine's outlier dummies (named AO<t> / LS<t> / SC<t>) are appended
    # to the coefficient vector but the Hessian-derived SE block covers
    # only the structural parameters.  Build `ses` aligned to `est` by
    # name; missing rows fall to NA so the downstream interval table
    # still has a sensible row per coefficient.
    ses     <- rep(NA_real_, length(est))
    names(ses) <- nm
    concIdx <- integer(0)
    if (!is.null(dim(cv))){
        cvNames <- rownames(cv)
        if (is.null(cvNames)) cvNames <- nm[seq_len(nrow(cv))]
        shared  <- intersect(nm, cvNames)
        cvDiag  <- diag(cv)
        ses[shared] <- sqrt(cvDiag[match(shared, cvNames)])
        concRows <- which(is.nan(cvDiag))
        concIdx  <- match(cvNames[concRows], nm)
        concIdx  <- concIdx[!is.na(concIdx)]
        if (length(concIdx) > 0 && is.finite(n) && n > 0){
            seConc       <- abs(est[concIdx]) * sqrt(2 / n)
            ses[concIdx] <- seConc
            cvRowConc <- match(nm[concIdx], cvNames)
            diag(cv)[cvRowConc] <- seConc ^ 2
            for (i in cvRowConc){
                other <- setdiff(seq_len(nrow(cv)), i)
                cv[i, other] <- 0
                cv[other, i] <- 0
            }
        }
    }

    a     <- (1 - level) / 2
    z     <- qnorm(c(a, 1 - a))
    lower <- est + ses * z[1]
    upper <- est + ses * z[2]

    # Coefficient table: Estimate / Std. Error / Lower / Upper only.  t
    # and p-values were dropped -- with finite-sample Hessian-derived SEs
    # the asymptotic z-test they imply gives a false sense of precision,
    # and the (Lower, Upper) interval already tells the same story.
    cmat <- cbind(Estimate     = est,
                  `Std. Error` = ses,
                  Lower        = lower,
                  Upper        = upper)
    rownames(cmat) <- nm

    # Inject the deterministic slope (if any) right after Level.  The slope
    # row has NA for SE / Lower / Upper -- print.summary.pts uses na.print="-".
    detSlope <- .pts_det_slope(object)
    if (!is.null(detSlope)){
        lvlIdx <- which(rownames(cmat) == "Level")
        if (length(lvlIdx) == 1L){
            slopeRow <- matrix(c(detSlope, NA_real_, NA_real_, NA_real_),
                               nrow = 1L,
                               dimnames = list("Slope", colnames(cmat)))
            cmat <- rbind(cmat[seq_len(lvlIdx), , drop = FALSE], slopeRow,
                          if (lvlIdx < nrow(cmat))
                              cmat[(lvlIdx + 1L):nrow(cmat), , drop = FALSE]
                          else cmat[integer(0), , drop = FALSE])
        }
    }

    # Variance proportions including Irregular so they sum to 1.
    isArma   <- grepl("^S?(AR|MA)\\(", nm)   # matches AR/MA + SAR/SMA
    isXreg   <- grepl("^Beta",       nm)
    isOutlier<- grepl("^(AO|LS|SC)[0-9]+$", nm)   # engine outlier dummies
    isDamp   <- nm == "Damping"
    isVar    <- !(isArma | isXreg | isDamp | isOutlier)
    varVals <- est[isVar]
    S       <- sum(varVals)
    props   <- if (length(varVals) > 0 && S > 0) varVals / S else varVals

    propSEs <- rep(NA_real_, length(varVals))
    names(propSEs) <- names(varVals)
    if (length(varVals) > 1 && !is.null(dim(cv)) && S > 0){
        varIdx <- which(isVar)
        if (max(varIdx) <= nrow(cv)){
            Sv <- cv[varIdx, varIdx, drop = FALSE]
            if (all(is.finite(Sv))){
                J  <- (diag(length(varVals)) - outer(rep(1, length(varVals)), props)) / S
                propVar <- diag(J %*% Sv %*% t(J))
                propSEs <- sqrt(pmax(0, propVar))
            }
        }
    }
    propMat <- cbind(Proportion  = props,
                     `Std. Error` = propSEs)
    rownames(propMat) <- names(props)

    # Trigonometric seasonality (only branch where the harmonic list is
    # interpretable): seasonal segment of the UC string is "equal".
    isTrig <- !is.null(object$modelUC) &&
              grepl("/equal/", object$modelUC, fixed = TRUE)

    out <- list(
        coefficients = cmat,
        proportions  = propMat,
        concentrated = nm[concIdx],
        sigma        = sigma(object),
        logLik       = as.numeric(logLik(object)),
        nobs         = n,
        nParam       = nparam(object),
        IC           = c(AIC  = AIC(object),
                         AICc = AICc(object),
                         BIC  = BIC(object),
                         BICc = BICc(object)),
        model        = object$model,
        modelUC      = object$modelUC,
        responseName = .pts_response_name(object),
        lambda       = object$lambda,
        lags         = object$lags,
        lagsAll      = object$lagsAll,
        isTrig       = isTrig,
        loss         = object$loss,
        lossValue    = object$lossValue,
        level        = level,
        call         = object$call,
        timeElapsed  = object$timeElapsed
    )
    class(out) <- c("summary.pts", "list")
    out
}

#' @export
print.summary.pts <- function(x, digits = 4, ...){
    fnName <- tryCatch(utils::tail(all.vars(x$call[[1]]), 1),
                       error = function(e) "pts")
    if (!nzchar(fnName)) fnName <- "pts"

    # --- Header block (mirrors smooth::print.summary.adam) ---
    cat("Model estimated using ", fnName, "() function: ", x$model, "\n", sep = "")
    cat("Response variable: ", x$responseName, "\n", sep = "")
    cat("Box-Cox lambda: ", format(x$lambda, digits = digits), "\n", sep = "")

    if (isTRUE(x$isTrig) && !is.null(x$lagsAll) && length(x$lagsAll) > 1){
        perStr <- paste(formatC(x$lagsAll, format = "f", digits = 1), collapse = " / ")
        cat("Harmonics: ", perStr, "\n", sep = "")
    }

    cat("Loss function type: ", x$loss, sep = "")
    if (!is.null(x$lossValue) && is.finite(x$lossValue))
        cat("; Loss function value: ", round(x$lossValue, digits), sep = "")
    cat("\n")

    # --- Coefficients table ---
    cat("\nCoefficients (", format(100 * x$level, digits = digits), "% CI):\n", sep = "")
    cmat <- x$coefficients
    cmat_print <- cmat
    cmat_print[, "Estimate"]   <- signif(cmat[, "Estimate"],   digits)
    cmat_print[, "Std. Error"] <- signif(cmat[, "Std. Error"], digits)
    cmat_print[, "Lower"]      <- signif(cmat[, "Lower"],      digits)
    cmat_print[, "Upper"]      <- signif(cmat[, "Upper"],      digits)
    rn <- rownames(cmat_print)
    if (length(x$concentrated) > 0)
        rn[rn %in% x$concentrated] <- paste0(rn[rn %in% x$concentrated], " (*)")
    rownames(cmat_print) <- rn
    print(cmat_print, na.print = "-")

    # --- Variance proportions ---
    if (!is.null(x$proportions) && nrow(x$proportions) > 0){
        cat("\nVariance proportions:\n")
        pmat_print <- x$proportions
        pmat_print[, "Proportion"] <- signif(x$proportions[, "Proportion"], digits)
        pmat_print[, "Std. Error"] <- signif(x$proportions[, "Std. Error"], digits)
        rn <- rownames(pmat_print)
        if (length(x$concentrated) > 0)
            rn[rn %in% x$concentrated] <- paste0(rn[rn %in% x$concentrated], " (*)")
        rownames(pmat_print) <- rn
        print(pmat_print, na.print = "-")
    }

    if (length(x$concentrated) > 0)
        cat("(*) concentrated out; Std. Error is the analytical Gaussian-MLE",
            "value sigma^2 * sqrt(2/n).\n")
    if ("Slope" %in% rownames(x$coefficients) &&
        !is.null(x$modelUC) && startsWith(x$modelUC, "td/"))
        cat("Slope is deterministic under the global trend (G);",
            "no variance, no Std. Error.\n")

    # --- Footer block ---
    cat("\nSample size: ", x$nobs, "\n", sep = "")
    cat("Number of estimated parameters: ", x$nParam, "\n", sep = "")
    cat("Number of degrees of freedom: ", x$nobs - x$nParam, "\n", sep = "")
    cat("Information criteria:\n")
    print(round(x$IC, digits))

    invisible(x)
}

#' @export
as.data.frame.summary.pts <- function(x, row.names = NULL,
                                       optional = FALSE, ...){
    df <- as.data.frame(unclass(x$coefficients), row.names = row.names,
                         optional = optional, ...)
    df$Parameter <- rownames(x$coefficients)
    df[, c("Parameter", "Estimate", "Std. Error", "Lower", "Upper")]
}
