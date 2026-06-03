# simulate.pts -- forward-simulate sample paths from the fitted PTS model.
# The new C++ "simulate" command propagates
#     a_{t+1} = T * a_t + R * eta,   eta ~ N(0, Q)
#     y_t    = Z * a_t + eps,        eps ~ N(0, H)
# starting from the terminal filtered state, then back-transforms each
# path with invBoxCox(., lambda) so the user sees original-scale series.
# Returns a ts matrix (h x nsim) of class c("pts.sim", "smooth.sim").

#' @export
simulate.pts <- function(object, nsim = 1, seed = NULL,
                         h = 10, ...){
    if (!is.numeric(nsim) || length(nsim) != 1 || nsim < 1)
        stop("`nsim` must be a positive integer.", call. = FALSE)
    if (!is.numeric(h) || length(h) != 1 || h < 1)
        stop("`h` must be a positive integer.", call. = FALSE)
    # Armadillo, when linked against R, delegates randn() to R's RNG and
    # ignores its own seed.  So reproducibility goes through R: set the
    # seed here, save / restore .Random.seed if it was untouched.
    if (!is.null(seed)){
        old <- if (exists(".Random.seed", envir = globalenv()))
                   get(".Random.seed", envir = globalenv())
               else NULL
        on.exit({
            if (is.null(old)) rm(".Random.seed", envir = globalenv())
            else assign(".Random.seed", old, envir = globalenv())
        }, add = TRUE)
        set.seed(seed)
    }

    args         <- .pts_forecast_inputs(object, h)
    args$nsim    <- as.integer(nsim)
    args$seed    <- 0L      # C++ set_seed is ignored; R RNG is the source of truth
    out          <- suppressWarnings(.pts_call_uc("simulate", args))

    paths <- as.matrix(out$simPaths)        # h x nsim, original scale
    # Anchor as a ts matrix starting one period after the last observation.
    if (is.ts(object$data)){
        fy <- stats::frequency(object$data)
        sy <- stats::start(object$data, frequency = fy)
        aux <- stats::ts(rep(NA_real_, length(object$data) + 1L),
                         start = sy, frequency = fy)
        paths <- stats::ts(paths, start = stats::end(aux), frequency = fy)
    }

    ret <- list(
        data    = paths,        # smooth.sim convention: $data is the simulated matrix
        model   = object$model,
        nsim    = as.integer(nsim),
        h       = as.integer(h),
        seed    = seed
    )
    class(ret) <- c("pts.sim", "smooth.sim")
    ret
}

#' @export
print.pts.sim <- function(x, digits = 4, ...){
    cat("PTS simulated paths: ", x$model, "\n", sep = "")
    cat("  nsim: ", x$nsim, "   h: ", x$h, "\n", sep = "")
    if (!is.null(x$seed)) cat("  seed: ", x$seed, "\n", sep = "")
    cat("\nFirst path, first ", min(6, x$h), " steps:\n", sep = "")
    print(round(as.numeric(x$data[seq_len(min(6, x$h)), 1]), digits))
    invisible(x)
}
