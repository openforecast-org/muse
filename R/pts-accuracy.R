# accuracy.pts -- delegate to greybox::measures() for the standard set of
# error metrics.  Supports two call shapes:
#   accuracy(m)                            -> uses m$holdout if available
#   accuracy(m, holdout = <vector>)        -> use the passed-in vector

#' @export
accuracy.pts <- function(object, holdout = NULL, ...){
    if (is.null(holdout)) holdout <- object$holdout
    if (is.null(holdout))
        stop("No holdout provided and the pts object does not carry one. ",
             "Either fit with `holdout = TRUE` or pass `holdout` explicitly.",
             call. = FALSE)
    h <- length(holdout)
    pred <- as.numeric(forecast(object, h = h)$mean)
    greybox::measures(holdout = as.numeric(holdout),
                      forecast = pred,
                      actual   = as.numeric(actuals(object)))
}

#' @export
accuracy.pts.forecast <- function(object, holdout = NULL, ...){
    if (is.null(holdout)) holdout <- object$model$holdout
    if (is.null(holdout))
        stop("No holdout provided and the underlying pts has none.",
             call. = FALSE)
    greybox::measures(holdout = as.numeric(holdout),
                      forecast = as.numeric(object$mean),
                      actual   = as.numeric(actuals(object$model)))
}
