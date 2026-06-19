# simulate.pts -- replay sample paths from the fitted PTS model starting
# at the initial state.  Mirrors smooth::simulate.adam's convention:
#     simulate(object, nsim, seed, obs = nobs(object), ...)
# The C++ "simulateInit" command propagates
#     a_{t+1} = T * a_t + R * eta,   eta ~ N(0, Q)
#     y_t    = Z * a_t + eps,        eps ~ N(0, H)
# starting from `betaAug` (the augmented KF's MLE of alpha_0), runs for `obs`
# steps, then back-transforms with invBoxCox(., lambda) so the user sees
# original-scale series anchored to the same start as `object$data`.
#
# For forward-from-end-of-sample simulation (used by
# forecast.pts(..., interval = "simulated")), see .pts_forecast_paths.

#' @export
simulate.pts <- function(object, nsim = 1, seed = NULL,
                         obs = nobs(object), ...){
    if (!is.numeric(nsim) || length(nsim) != 1 || nsim < 1)
        stop("`nsim` must be a positive integer.", call. = FALSE)
    if (!is.numeric(obs) || length(obs) != 1 || obs < 1)
        stop("`obs` must be a positive integer.", call. = FALSE)
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

    args      <- .pts_forecast_inputs(object, as.integer(obs))
    args$nsim <- as.integer(nsim)
    args$seed <- 0L     # C++ set_seed ignored; R RNG drives reproducibility
    # The engine runs its smoother internally for "simulateInit" and uses
    # the smoothed alpha_{t=1} as the seed (see musecore.h dispatch).
    out       <- suppressWarnings(.pts_call_uc("simulateInit", args))

    paths <- as.matrix(out$simPaths)   # obs x nsim, original scale
    # Anchor: paths share the start of object$data and run for `obs` steps.
    if (is.ts(object$data)){
        fy <- stats::frequency(object$data)
        sy <- stats::start(object$data, frequency = fy)
        paths <- stats::ts(paths, start = sy, frequency = fy)
    }

    ret <- list(
        data  = paths,
        model = object$model,
        nsim  = as.integer(nsim),
        obs   = as.integer(obs),
        seed  = seed
    )
    class(ret) <- c("pts.sim", "smooth.sim")
    ret
}

# .pts_forecast_paths -- forward simulation from the terminal state.
# Internal helper for forecast.pts(..., interval = "simulated").
# Same engine as simulate.pts but uses the "simulate" command (= aEnd
# seed) and anchors the result one period after the last observation.
.pts_forecast_paths <- function(object, nsim, h, seed = NULL){
    if (!is.null(seed)) set.seed(seed)
    args      <- .pts_forecast_inputs(object, as.integer(h))
    args$nsim <- as.integer(nsim)
    args$seed <- 0L
    out       <- suppressWarnings(.pts_call_uc("simulate", args))
    paths     <- as.matrix(out$simPaths)
    if (is.ts(object$data)){
        fy <- stats::frequency(object$data)
        sy <- stats::start(object$data, frequency = fy)
        aux <- stats::ts(rep(NA_real_, length(object$data) + 1L),
                         start = sy, frequency = fy)
        paths <- stats::ts(paths, start = stats::end(aux), frequency = fy)
    }
    paths
}

#' @export
print.pts.sim <- function(x, digits = 4, ...){
    cat("PTS simulated paths: ", x$model, "\n", sep = "")
    cat("  nsim: ", x$nsim, "   obs: ", x$obs, "\n", sep = "")
    if (!is.null(x$seed)) cat("  seed: ", x$seed, "\n", sep = "")
    cat("\nFirst path, first ", min(6, x$obs), " steps:\n", sep = "")
    print(round(as.numeric(x$data[seq_len(min(6, x$obs)), 1]), digits))
    invisible(x)
}
