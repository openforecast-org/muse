# Re-export the `forecast` generic from generics and bring smooth/greybox
# generics into the muse namespace so the new S3 methods register against
# them.  These three packages are already declared in DESCRIPTION; this
# block only imports the names we need.

#' @importFrom generics forecast accuracy
#' @export
generics::forecast

#' @importFrom greybox actuals errorType nparam extractScale extractSigma
#' @importFrom greybox AICc BICc pointLik outlierdummy measures
#' @importFrom smooth modelType modelName lags orders pls
#' @importFrom stats sigma confint simulate update rstandard rstudent qnorm pnorm dnorm sd
NULL
