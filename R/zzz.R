# Re-export the forecast() generic from the generics package so that
# forecast(object, h) dispatches to forecast.pts() without users having to
# attach forecast / smooth first.

#' @importFrom generics forecast
#' @export
generics::forecast
