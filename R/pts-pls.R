# pls.pts -- Prediction Least Squares: sum of squared h-step-ahead
# forecast errors on a held-out section.  Uses the cached forecast_args
# so changing h is cheap (no re-estimation).
#
# When holdout is omitted, we use object$holdout (only available when
# pts(y, holdout = TRUE) was used at fit time).

#' @export
pls.pts <- function(object, holdout = NULL, ...){
    if (is.null(holdout)) holdout <- object$holdout
    if (is.null(holdout))
        stop("pls.pts needs a holdout vector or a model fitted with holdout = TRUE.",
             call. = FALSE)
    h    <- length(holdout)
    pred <- as.numeric(forecast(object, h = h)$mean)
    sum((as.numeric(holdout) - pred) ^ 2, na.rm = TRUE)
}
