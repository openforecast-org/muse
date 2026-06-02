#' @title MSOEsetup
#' @description Sets up MSOE general univariate models
#'
#' @details See help of \code{MSOE}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{periods} should be supplied compulsorily (see below).
#' @param u a matrix of external regressors included only in the observation equation.
#' (it may be either a numerical vector or a time series object). If the output wanted
#' to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. It is a single string indicating the type of
#' model for each component. It allows two formats "trend/seasonal/irregular" or
#' "trend/cycle/seasonal/irregular". The possibilities available for each component are:
#' \itemize{
#' \item Trend: ? / none / rw / irw / llt / dt / td;
#'
#' \item Seasonal: ? / none / equal / different;
#'
#' \item Irregular: ? / none / arma(0, 0) / arma(p, q) - with p and q integer positive orders;
#'
#' \item Cycles: ? / none / combination of positive or negative numbers. Positive numbers fix
#' the period of the cycle while negative values estimate the period taking as initial
#' condition the absolute value of the period supplied. Several cycles with positive or negative values are possible
#' and if a question mark is included, the model test for the existence of the cycles
#' specified. The following are valid examples with different meanings: 48, 48?, -48, -48?,
#' 48+60, -48+60, -48-60, 48-60, 48+60?, -48+60?, -48-60?, 48-60?.
#' }
#' @param outlier critical level of outlier tests. If NA it does not carry out any
#' outlier detection (default). A positive value indicates the critical minimum
#' t test for outlier detection in any model during identification. Three types of outliers are
#' identified, namely Additive Outliers (AO), Level Shifts (LS) and Slope Change (SC).
#' @param stepwise stepwise identification procedure (TRUE / FALSE).
#' @param tTest augmented Dickey Fuller test for unit roots used in stepwise algorithm (TRUE / FALSE).
#' The number of models to search for is reduced, depending on the result of this test.
#' @param p0 initial parameter vector for optimisation search.
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param lambda Box-Cox transformation lambda, NULL for automatic estimation
#' @param criterion information criterion for identification ("aic", "bic" or "aicc").
#' @param periods vector of fundamental period and harmonics required.
#' @param verbose intermediate results shown about progress of estimation (TRUE / FALSE).
#' @param arma check for arma models for irregular components (TRUE / FALSE).
#' @param TVP vector of zeros and ones to indicate TVP parameters.
#' @param trendOptions trend models to select amongst (e.g., "rw/llt").
#' @param seasonalOptions seasonal models to select amongst (e.g., "none/differentt").
#' @param irregularOptions irregular models to select amongst (e.g., "none/arma(0,1)").
#'
#' @author Diego Pedregal & Ivan Svetunkov
#'
#' @return An object of class \code{MSOE}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{MSOE} object as specified in what follows (function
#'         \code{MSOE} fills in all of them at once):
#'
#' After running \code{MSOEmodel} or \code{MSOEestim}:
#' \itemize{
#' \item p:        Estimated parameters
#' \item v:        Estimated innovations (white noise in correctly specified models)
#' \item yFor:     Forecasted values of output
#' \item yForV:    Variance of forecasts
#' \item criteria: Value of criteria for estimated model
#' \item iter:     Number of iterations in estimation
#' \item grad:     Gradient at estimated parameters
#' \item covp:     Covariance matrix of parameters
#' }
#'
#' After running \code{MSOEvalidate}:
#' \itemize{
#' \item table: Estimation and validation table
#' }
#'
#' After running \code{MSOEcomponents}:
#' \itemize{
#' \item comp:      Estimated components in matrix form
#' \item compV:     Estimated components variance in matrix form
#' }
#'
#' After running \code{MSOEfilter}, \code{MSOEsmooth}:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Estimated fitted values variance
#' \item a:     State estimates
#' \item P:     Variance of state estimates
#' \item aFor:  Forecasts of states
#' \item PFor:  Forecasts of states variances
#' }
#'
#' Standard methods applicable to MSOE objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}}, \code{\link{MSOEsmooth}},
#'          \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' y <- log(AirPassengers)
#' m1 <- MSOEsetup(y)
#' m1 <- MSOEsetup(y, outlier = 4)
#' m1 <- MSOEsetup(y, model = "llt/equal/arma(0,0)")
#' m1 <- MSOEsetup(y, model = "?/?/?/?")
#' m1 <- MSOEsetup(y, model = "llt/?/equal/?", outlier = 4)
#' }
#' @keywords internal
#' @noRd
MSOEsetup = function(y, u = NULL, model = "?/none/?/?", h = 9999, lambda = 1, outlier = 9999, tTest = FALSE, criterion = "aic",
                   periods = NA, verbose = FALSE, stepwise = FALSE, p0 = -9999.9, arma = FALSE,
                   TVP = NULL, trendOptions = "rw/llt/srw/td", seasonalOptions = "none/linear/equal", irregularOptions = "arma(0,0)"){
    # Converting u vector to matrix
    if (length(size(u)) == 1 && size(u) > 0){
        u = matrix(u, 1, length(u))
    }
    if (length(size(u)) == 2 && size(u)[1] > size(u)[2])
        u = t(u)
    if (is.null(TVP) && !is.null(u))
        TVP = matrix(0, nrow(u))
    if (is.null(TVP))
        TVP = -9999.99
    if (is.null(u)){
        u = matrix(0, 1, 2)
    }
    if (is.null(lambda))
        lambda = 9999.9
    # Checking periods
    if (is.ts(y) && any(is.na(periods)) && frequency(y) > 1){
        periods = frequency(y) / (1 : floor(frequency(y) / 2))
    } else if (is.ts(y) && any(is.na(periods))){
        periods = 1
    } else if (is.ts(y) && any(is.infinite(periods))){
        periods = 1
    } else if (!is.ts(y) && any(is.na(periods))){
        stop("Input \"periods\" should be supplied!!")
    }
    rhos = rep(1, length(periods))
    out = list(y = y,
               u = u,
               model = model,
               h = h,
               lambda = lambda,
               # Other less important
               arma =  arma,
               outlier = -abs(outlier),
               tTest = tTest,
               criterion = criterion,
               periods = periods,
               rhos = rhos,
               verbose = verbose,
               stepwise = stepwise,
               p0 = p0,
               criteria = NA,
               TVP = TVP,
               trendOptions = trendOptions,
               seasonalOptions = seasonalOptions,
               irregularOptions = irregularOptions,
               # Outputs
               comp = NA,
               compPlus = NA,
               compMinus = NA,
               p = NA,
               covp = NA,
               grad = NA,
               v= NA,
               yFit = NA,
               yFor = NA,
               yFitV = NA,
               yForV = NA,
               a = NA,
               P = NA,
               eta = NA,
               eps = NA,
               table = NA,
               iter = 0,
               # Other variables
               hidden = list(d_t = 0,
                             estimOk = "Not estimated",
                             objFunValue = 0,
                             innVariance = 1,
                             nonStationaryTerms = NA,
                             ns = NA,
                             nPar = NA,
                             harmonics = 0,
                             constPar = NA,
                             typePar = NA,
                             cycleLimits = NA,
                             typeOutliers = matrix(-1, 1, 2),
                             truePar = NA,
                             beta = NA,
                             betaV = NA,
                             seas = frequency(y),
                             MSOE = FALSE,
                             PTSnames = TRUE))
    return(structure(out, class = "MSOE"))
}
#' @title MSOE
#' @description Runs all relevant functions for MSOE modelling
#'
#' @details \code{MSOE} is a function for modelling and forecasting univariate
#' time series according to Unobserved Components models (MSOE).
#' It sets up the model with a number of control variables that
#' govern the way the rest of functions in the package work. It also estimates
#' the model parameters by Maximum Likelihood, forecasts the data, performs smoothing,
#' estimates model disturbances, estimates components and shows statistical diagnostics.
#' Standard methods applicable to MSOE objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @inheritParams MSOEsetup
#'
#' @template authors
#'
#' @return An object of class \code{MSOE}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{MSOE} object as specified in what follows (function
#'         \code{MSOE} fills in all of them at once):
#'
#' After running \code{MSOEmodel} or \code{MSOEestim}:
#' \itemize{
#' \item p:        Estimated parameters
#' \item v:        Estimated innovations (white noise in correctly specified models)
#' \item yFor:     Forecasted values of output
#' \item yForV:    Forecasted values +- one standard error
#' \item criteria: Value of criteria for estimated model
#' \item iter:     Number of iterations in estimation
#' \item grad:     Gradient at estimated parameters
#' \item covp:     Covariance matrix of parameters
#' }
#'
#' After running \code{MSOEvalidate}:
#' \itemize{
#' \item table: Estimation and validation table
#' }
#'
#' After running \code{MSOEcomponents}:
#' \itemize{
#' \item comp:  Estimated components in matrix form
#' \item compV: Estimated components variance in matrix form
#' }
#'
#' After running \code{MSOEfilter}, \code{MSOEsmooth} or  \code{MSOEdisturb}:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates
#' \item aFor:  Forecasts of states
#' \item PFor:  Forecasts of states variances
#' }
#'
#' After running \code{MSOEdisturb}:
#' \itemize{
#' \item eta: State perturbations estimates
#' \item eps: Observed perturbations estimates
#' }
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}}, \code{\link{MSOEsmooth}},
#'          \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' y <- log(AirPassengers)
#' m1 <- MSOE(y)
#' m1 <- MSOE(y, model = "llt/different/arma(0,0)")
#' }
#' @keywords internal
#' @noRd
MSOE = function(sys) {
    output = UCompC("all", sys$y, sys$u, sys$model, sys$h, sys$lambda, sys$outlier, sys$tTest,
                    sys$criterion, sys$periods, sys$rhos, sys$verbose, sys$stepwise, sys$p0, sys$arma, sys$TVP,
                    sys$hidden$seas, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    sys$lambda = output$lambda
    if (output$model == "error"){
        sys$model = "error"
        return(sys);
    }
    # MSOE used to leave sys$model as the user's "?" template; copy the
    # resolved model string back from C++ so downstream code (PTS, pts)
    # sees the actual fitted model.
    sys$model = output$model
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        aux = ts(matrix(NA, length(sys$y) + 1, 1), sY, frequency = fY)
        if (length(output$yFor > 0)){
            sys$yFor = ts(output$yFor, end(aux), frequency = fY)
            sys$yForV = ts(output$yForV, end(aux), frequency = fY)
        }
    } else {
        if (length(output$yFor > 0)){
            sys$yFor = output$yFor
            sys$yForV = output$yForV
        }
    }
    # validaton
    sys$table = output$table
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        aux = ts(matrix(NA, length(sys$y) - length(output$v) + 1, 1), sY, frequency = fY)
        sys$v = ts(output$v, end(aux), frequency = fY)
    } else {
        sys$v = output$v
    }
    sys$covp = output$covp
    sys$criteria = as.numeric(output$criteria)
    if (length(sys$criteria) == 4L)
        names(sys$criteria) = c("logLik", "AIC", "BIC", "AICc")
    sys$p = as.vector(output$coef)
    nPar = length(sys$p)
    rownames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
    colnames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
    names(sys$p) = output$parNames[1 : nPar]
    sys$comp = output$comp
    sys$compV = output$compV
    m = output$m  # + nCycles
    if (dim(sys$u)[1] == 1 && dim(sys$u)[2] == 2){
        k = 0
    } else {
        k = dim(sys$u)[1]
    }
    nCycles = m - k - 4
    # Re-building matrices to their original sizes
    n = length(sys$comp) / m
    if (is.ts(sys$y)){
        sys$comp = ts(t(matrix(sys$comp, m, n)), start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
        sys$compV = ts(t(matrix(sys$compV, m, n)), start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
    } else {
        sys$comp = t(matrix(sys$comp, m, n))
        sys$compV = t(matrix(sys$compV, m, n))
    }
    if (length(size(sys$comp)) == 1){
        if (is.ts(sys$y)){
            sys$comp = ts(matrix(sys$comp, n + sys$h, 1), start = start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
            sys$compV = ts(matrix(sys$compV, n + sys$h, 1), start = start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
        } else {
            sys$comp = matrix(sys$comp, n + sys$h, 1)
            sys$compV = matrix(sys$compV, n + sys$h, 1)
        }
    }
    names = strsplit(output$compNames, "/")
    colnames(sys$comp) = names[[1]]
    colnames(sys$compV) = names[[1]]
    return(sys)
}
#' @title MSOEestim
#' @description Estimates and forecasts MSOE models
#'
#' @details \code{MSOEestim} estimates and forecasts a time series using an
#' MSOE model.
#' The optimization method is a BFGS quasi-Newton algorithm with a
#' backtracking line search using Armijo conditions.
#' Parameter names in output table are the following:
#' \itemize{
#' \item Damping:   Damping factor for DT trend.
#' \item Level:     Variance of level disturbance.
#' \item Slope:     Variance of slope disturbance.
#' \item Rho(#):    Damping factor of cycle #.
#' \item Period(#): Estimated period of cycle #.
#' \item Var(#):    Variance of cycle #.
#' \item Seas(#):   Seasonal harmonic with period #.
#' \item Irregular: Variance of irregular component.
#' \item AR(#):     AR parameter of lag #.
#' \item MA(#):     MA parameter of lag #.
#' \item AO#:       Additive outlier in observation #.
#' \item LS#:       Level shift outlier in observation #.
#' \item SC#:       Slope change outlier in observation #.
#' \item Beta(#):   Beta parameter of input #.
#' \item Cnst:      Constant.
#' }
#'
#' Standard methods applicable to MSOEomp objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{MSOE} created with \code{MSOE}
#'
#' @return The same input object with the appropriate fields
#' filled in, in particular:
#' \itemize{
#' \item p:        Estimated transformed parameters
#' \item v:        Estimated innovations (white noise in correctly specified models)
#' \item yFor:     Forecast values of output
#' \item yForV:    Forecasted values variance
#' \item criteria: Value of criteria for estimated model
#' \item covp:     Covariance matrix of estimated transformed parameters
#' \item grad:     Gradient of log-likelihood at the optimum
#' \item iter:     Estimation iterations
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}},
#'          \code{\link{MSOEsmooth}}, \code{\link{MSOEcomponents}}
#'
#'
#' @examples
#' \dontrun{
#' # m1 <- MSOEsetup(log(AirPassengers))
#' # m1 <- MSOEestim(m1)
#' }
#' @keywords internal
#' @noRd
MSOEestim = function(sys){
    output = UCompC("estimate", sys$y, sys$u, sys$model, sys$h, sys$lambda, sys$outlier, sys$tTest,
                    sys$criterion, sys$periods, sys$rhos, sys$verbose, sys$stepwise, sys$p0, sys$arma, sys$TVP,
                    sys$hidden$seas, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    sys$lambda = output$lambda
    if (output$model == "error"){
        sys$model = "error"
        return(sys);
    }
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        aux = ts(matrix(NA, length(sys$y) + 1, 1), sY, frequency = fY)
        if (length(output$yFor > 0)){
            sys$yFor = ts(output$yFor, end(aux), frequency = fY)
            sys$yForV = ts(output$yForV, end(aux), frequency = fY)
        }
    } else {
        if (length(output$yFor > 0)){
            sys$yFor = output$yFor
            sys$yForV = output$yForV
        }
    }
    return(sys)
}

