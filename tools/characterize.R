#!/usr/bin/env Rscript
# Characterization (golden-master) harness for the dead-code cleanup.
#
# Fits a broad battery of pts() configurations and exercises every method,
# serialising a deterministic numeric snapshot to an RDS.  Run once before
# the cleanup to record a baseline, again after, and diff:
#
#   Rscript tools/characterize.R record  /tmp/muse_baseline.rds
#   ...cleanup...
#   Rscript tools/characterize.R check   /tmp/muse_baseline.rds
#
# "check" recomputes the snapshot and compares field-by-field, reporting the
# max absolute numeric difference anywhere.  A clean dead-code removal must
# produce an identical snapshot (simulation paths are captured under fixed
# seeds; if the engine RNG stream is perturbed they are reported separately).

suppressMessages(devtools::load_all(".", quiet = TRUE))

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1) args[[1]] else "record"
path <- if (length(args) >= 2) args[[2]] else "/tmp/muse_baseline.rds"

## ---- datasets (positive; some with a strong trend / seasonality) ----
set.seed(1)
datasets <- list(
    air      = AirPassengers,
    air_log  = log(AirPassengers),
    bjsales  = BJsales,
    co2      = co2,
    nottem   = nottem,
    short    = ts(abs(rnorm(30, 100, 10)) + 1:30, frequency = 12),
    trendy   = ts(pmax(1, round(50 + 2 * (1:60) + 15 * sin(2*pi*(1:60)/12)
                                 + rnorm(60, 0, 5))), frequency = 12)
)

## ---- model specs spanning power / trend / seasonal / arma / ic ----
specs <- list(
    list(model = "ZZZ"),
    list(model = "ZZZ", ic = "AIC"),
    list(model = "ZZZ", ic = "BIC"),
    list(model = "ZZZ", ic = "AICc"),
    list(model = "1NN"),
    list(model = "1LN"),
    list(model = "1LT"),
    list(model = "1DT"),
    list(model = "1GT"),
    list(model = "1ND"),
    list(model = "0LT"),
    list(model = "0.5LT"),
    list(model = "2LT"),
    list(model = "1ZN"),
    list(model = "1ZZ"),
    list(model = "ZNN"),
    list(model = "1LT", orders = list(ar = 1, ma = 0)),
    list(model = "1LT", orders = list(ar = 0, ma = 1)),
    list(model = "1LT", orders = list(ar = 1, ma = 1)),
    list(model = "1LT", orders = list(ar = 2, ma = 2, select = TRUE)),
    list(model = "ZZZ", orders = list(ar = 1, ma = 1, select = TRUE))
)

`%||%` <- function(a, b) if (is.null(a)) b else a

num <- function(x) {
    if (is.null(x)) return(NULL)
    x <- suppressWarnings(as.numeric(x))
    x
}

## ---- capture one (dataset, spec) cell ----
capture_cell <- function(y, spec) {
    h <- 8L
    call_args <- c(list(data = y, h = h, holdout = TRUE), spec)
    m <- tryCatch(do.call(pts, call_args), error = function(e) e)
    if (inherits(m, "error")) return(list(error = conditionMessage(m)))

    out <- list()
    out$model      <- m$model
    out$modelUC    <- m$modelUC
    out$lambda     <- num(m$lambda)
    out$nParam     <- num(m$nParam)
    out$logLik     <- num(stats::logLik(m))
    out$AIC        <- num(AIC(m))
    out$BIC        <- num(BIC(m))
    out$AICc       <- num(greybox::AICc(m))
    out$BICc       <- num(greybox::BICc(m))
    out$nobs       <- num(stats::nobs(m))
    out$sigma      <- num(sigma(m))
    out$coef       <- num(coef(m));        names(out$coef) <- names(coef(m))
    out$vcov_diag  <- num(diag(as.matrix(vcov(m))))
    out$fitted     <- num(fitted(m))
    out$residuals  <- num(residuals(m))
    out$comp       <- num(as.matrix(m$comp))
    out$states     <- num(as.matrix(m$states))
    out$rstandard  <- num(rstandard(m))
    out$rstudent   <- num(rstudent(m))
    out$pointLik   <- num(pointLik(m))
    out$confint    <- num(as.matrix(suppressWarnings(confint(m))))
    out$orders     <- num(unlist(orders(m)))
    out$lags       <- num(lags(m))
    out$modelType  <- as.character(modelType(m))
    out$errorType  <- as.character(errorType(m))

    # forecast + intervals (deterministic types)
    fc <- tryCatch(forecast(m, h = h, interval = "prediction",
                            level = c(0.8, 0.95)),
                   error = function(e) e)
    if (!inherits(fc, "error")) {
        out$fc_mean  <- num(fc$mean)
        out$fc_lower <- num(as.matrix(fc$lower))
        out$fc_upper <- num(as.matrix(fc$upper))
    }
    fcc <- tryCatch(forecast(m, h = h, interval = "confidence", level = 0.95),
                    error = function(e) e)
    if (!inherits(fcc, "error")) {
        out$fcc_lower <- num(as.matrix(fcc$lower))
        out$fcc_upper <- num(as.matrix(fcc$upper))
    }

    # accuracy on the holdout
    acc <- tryCatch(num(accuracy(m)), error = function(e) NULL)
    out$accuracy <- acc

    # simulate (fixed seed) — capture path summary stats, robust to RNG drift
    sm <- tryCatch({
        set.seed(2024)
        s <- simulate(m, nsim = 20, seed = 2024)
        as.matrix(s$data)
    }, error = function(e) NULL)
    if (!is.null(sm)) {
        out$sim_mean <- num(colMeans(sm))
        out$sim_sd   <- num(apply(sm, 2, stats::sd))
    }
    out
}

## ---- build full snapshot ----
build_snapshot <- function() {
    snap <- list()
    for (dn in names(datasets)) {
        for (i in seq_along(specs)) {
            key <- paste0(dn, "::", i, "::",
                          specs[[i]]$model %||% "?")
            snap[[key]] <- capture_cell(datasets[[dn]], specs[[i]])
        }
    }
    snap
}

## ---- diff two snapshots ----
max_abs_diff <- function(a, b) {
    if (is.null(a) && is.null(b)) return(0)
    if (is.null(a) || is.null(b)) return(Inf)
    if (is.character(a) || is.character(b))
        return(if (identical(as.character(a), as.character(b))) 0 else Inf)
    a <- suppressWarnings(as.numeric(a)); b <- suppressWarnings(as.numeric(b))
    if (length(a) != length(b)) return(Inf)
    d <- abs(a - b)
    d[is.na(d) & is.na(a) & is.na(b)] <- 0   # NaN==NaN ok
    if (all(is.na(d))) return(0)
    max(d, na.rm = TRUE)
}

compare <- function(new, old) {
    keys <- union(names(new), names(old))
    worst <- 0; report <- character()
    sim_fields <- c("sim_mean", "sim_sd")
    sim_worst <- 0
    for (k in keys) {
        nc <- new[[k]]; oc <- old[[k]]
        if (is.null(nc) || is.null(oc)) {
            report <- c(report, sprintf("  [%s] MISSING in %s", k,
                                        if (is.null(nc)) "new" else "old"))
            worst <- Inf; next
        }
        flds <- union(names(nc), names(oc))
        for (f in flds) {
            d <- max_abs_diff(nc[[f]], oc[[f]])
            if (f %in% sim_fields) { sim_worst <- max(sim_worst, d); next }
            if (d > 1e-8) {
                report <- c(report, sprintf("  [%s] $%s  maxdiff=%.3g", k, f, d))
                worst <- max(worst, d)
            }
        }
    }
    list(worst = worst, sim_worst = sim_worst, report = report)
}

if (mode == "record") {
    snap <- build_snapshot()
    saveRDS(snap, path)
    n_ok  <- sum(vapply(snap, function(x) is.null(x[["error"]]), logical(1)))
    n_err <- length(snap) - n_ok
    cat(sprintf("Recorded %d cells (%d fitted, %d errored) -> %s\n",
                length(snap), n_ok, n_err, path))
} else {
    old <- readRDS(path)
    new <- build_snapshot()
    res <- compare(new, old)
    cat(sprintf("Cells: %d (baseline) vs %d (current)\n",
                length(old), length(new)))
    cat(sprintf("Deterministic max abs diff: %.3g\n", res$worst))
    cat(sprintf("Simulation summary max abs diff: %.3g\n", res$sim_worst))
    if (length(res$report)) {
        cat("DIFFERENCES:\n"); cat(res$report, sep = "\n"); cat("\n")
    }
    if (res$worst <= 1e-8) cat("RESULT: PASS (deterministic outputs identical)\n")
    else { cat("RESULT: FAIL (regression detected)\n"); quit(status = 1) }
}
