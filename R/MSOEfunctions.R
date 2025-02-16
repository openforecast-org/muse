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
#' @rdname MSOEsetup
#' @export
MSOEsetup = function(y, u = NULL, model = "?/none/?/?", h = 9999, lambda = 1, outlier = 9999, tTest = FALSE, criterion = "aic",
                   periods = NA, verbose = FALSE, stepwise = FALSE, p0 = -9999.9, arma = FALSE,
                   TVP = NULL, trendOptions = "rw/llt/srw/dt", seasonalOptions = "none/linear/equal", irregularOptions = "arma(0,0)"){
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
                             MSOE = TRUE,
                             PTSnames = TRUE))
    return(structure(out, class = "MSOE"))
}

#' @title MSOEmodel
#' @description Estimates and forecasts MSOE general univariate models
#'
#' @details \code{MSOEmodel} is a function for modelling and forecasting univariate
#' time series according to Unobserved Components models (MSOE).
#' It sets up the model with a number of control variables that
#' govern the way the rest of functions in the package work. It also estimates
#' the model parameters by Maximum Likelihood and forecasts the data.
#' Standard methods applicable to MSOE objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @inheritParams MSOEsetup
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
#' After running \code{MSOEfilter}, \code{MSOEsmooth}:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates
#' \item aFor:  Forecasts of states
#' \item PFor:  Forecasts of states variances
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}}, \code{\link{MSOEsmooth}},
#'          \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' y <- log(AirPassengers)
#' m1 <- MSOEmodel(y)
#' m1 <- MSOEmodel(y, model = "llt/equal/arma(0,0)")
#' }
#' @rdname MSOEmodel
#' @export
MSOEmodel = function(y, u = NULL, model = "?/none/?/?", h = 9999, lambda = 1, outlier = 9999, tTest = FALSE, criterion = "aic",
                   periods = NA, verbose = FALSE, stepwise = FALSE, p0 = -9999.9, arma = TRUE,
                   TVP = NULL, trendOptions = "rw/llt/srw/dt", seasonalOptions = "none/linear/equal", irregularOptions = "arma(0,0)"){
    m = MSOEsetup(y, u, model, h, lambda, outlier, tTest, criterion,
                periods, verbose, stepwise, p0, arma,
                TVP, trendOptions, seasonalOptions, irregularOptions)
    m = MSOEestim(m)
    return(m)
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
#' @rdname MSOE
#' @export
MSOE = function(y, u = NULL, model = "?/none/?/?", h = 9999, lambda = 1, outlier = 9999, tTest = FALSE, criterion = "aic",
              periods = NA, verbose = FALSE, stepwise = FALSE, p0 = -9999.9, arma = TRUE,
              TVP = NULL, trendOptions = "rw/llt/srw/dt", seasonalOptions = "none/linear/equal", irregularOptions = "arma(0,0)"){
    m = MSOEsetup(y, u, model, h, lambda, outlier, tTest, criterion,
                periods, verbose, stepwise, p0, arma,
                TVP, trendOptions, seasonalOptions, irregularOptions)
    m = MSOEestim(m)
    if (m$model == "error")
        return(m)
    m = MSOEvalidate(m, verbose)
    # m = MSOEdisturb(m)
    m = MSOEsmooth(m)
    m = MSOEcomponents(m)
    return(m)
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
#' @rdname MSOEestim
#' @export
MSOEestim = function(sys){
    sys$table = NA
    sys$hidden$constPar = NA
    # Estimation
    rubbish = c(sys$hidden$d_t, sys$hidden$innVariance, sys$hidden$objFunValue, TRUE,
                sys$outlier, sys$arma, sys$iter, sys$hidden$seas, sys$lambda,
                sys$hidden$MSOE, sys$hidden$PTSnames)
    rubbish2 = cbind(sys$grad, sys$hidden$constPar, sys$hidden$typePar)
    rubbish3 = cbind(sys$hidden$ns, sys$hidden$nPar)
    if (is.ts(sys$y)){
        y = as.numeric(sys$y)
    } else {
        y = sys$y
    }
    if (is.ts(sys$u)){
        u = as.numeric(sys$u)
    } else {
        u = sys$u
    }
    nu = dim(u)[2]
    kInitial = dim(u)[1]
    if (nu == 2){
        nu = length(sys$y) + sys$h
        kInitial = 0
    }
    output = MSOEc("estimate", y, u, sys$model, sys$periods, sys$rhos,
                    sys$h, sys$tTest, sys$criterion, sys$hidden$truePar, rubbish2, rubbish, sys$verbose,
                    sys$stepwise, sys$hidden$estimOk, sys$p0, sys$v, sys$yFitV,
                    sys$hidden$nonStationaryTerms, rubbish3, sys$hidden$harmonics,
                    as.vector(sys$criteria), sys$hidden$cycleLimits,
                    cbind(sys$hidden$beta, sys$hidden$betaV), sys$hidden$typeOutliers,
                    sys$TVP, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
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
    # Convert to R list
    #sys$p = output$p[, 2]
    sys$hidden$truePar = output$p[, 1]
    sys$p0 = output$p0
    if (grepl("?", sys$model, fixed = TRUE)){
        sys$model = output$model
    }
    n = length(sys$hidden$truePar)
    rubbish2 = matrix(output$rubbish2, n, 3)
    sys$grad = rubbish2[, 1]
    sys$hidden$constPar = rubbish2[, 2]
    sys$hidden$typePar = rubbish2[, 3]
    sys$hidden$cycleLimits = matrix(output$cycleLimits,
                                    length(output$cycleLimits) / 2, 2)
    sys$hidden$d_t = output$rubbish[1]
    sys$hidden$innVariance = output$rubbish[2]
    sys$hidden$objFunValue = output$rubbish[3]
    sys$iter = output$rubbish[6]
    sys$h = output$rubbish[7]
    sys$lambda = output$rubbish[8]
    betas = matrix(output$betas, length(output$betas) / 2, 2)
    sys$hidden$beta = betas[, 1]
    sys$hidden$betaV = betas[, 2]
    sys$periods = output$periods
    sys$rhos = output$rhos
    sys$hidden$estimOk = output$estimOk
    sys$hidden$nonStationaryTerms = output$nonStationaryTerms
    rubbish3 = matrix(output$rubbish3, 7, 2)
    sys$hidden$ns = rubbish3[, 1]
    sys$hidden$nPar = rubbish3[, 2]
    sys$hidden$harmonics = output$harmonics
    criteria = output$criteria;
    sys$criteria = matrix(criteria, 1, 4)
    colnames(sys$criteria) = c("LLIK", "AIC", "BIC", "AICc")
    sys$u = output$u
    if (!is.na(sys$outlier) && !is.null(u)){
        nu = length(sys$y) + sys$h;
        k = length(output$u) / nu
        nOut = k - kInitial
        if (nOut > 0){
            sys$u = matrix(output$u, k, nu)
            sys$hidden$typeOutliers = output$typeOutliers
        }
    }
    return(sys)
}

#' @title filter_
#' @description Auxiliar function of \code{MSOE} package.
#'
#' @param sys reserved input
#' @param command reserved input
#'
#' @template authors
#'
#' @noRd
filter_ = function(sys, command){
    if (is.ts(sys$y)){
        y = as.numeric(sys$y)
    } else {
        y = sys$y
    }
    if (is.ts(sys$u)){
        u = as.numeric(sys$u)
    } else {
        u = sys$u
    }
    rubbish = c(sys$hidden$d_t, sys$hidden$innVariance, sys$hidden$objFunValue, TRUE,
                sys$outlier, sys$arma, sys$iter, sys$hidden$seas, sys$lambda,
                sys$hidden$MSOE, sys$hidden$PTSnames)
    rubbish2 = cbind(sys$grad, sys$hidden$constPar, sys$hidden$typePar)
    rubbish3 = cbind(sys$hidden$ns, sys$hidden$nPar)
    output = MSOEc(command, y, u, sys$model, sys$periods, sys$rhos,
                    sys$h, sys$tTest, sys$criterion, sys$hidden$truePar, rubbish2, rubbish, sys$verbose,
                    sys$stepwise, sys$hidden$estimOk, sys$p0, sys$v, sys$yFitV,
                    sys$hidden$nonStationaryTerms, rubbish3, sys$hidden$harmonics,
                    as.vector(sys$criteria), sys$hidden$cycleLimits,
                    cbind(sys$hidden$beta, sys$hidden$betaV), sys$hidden$typeOutliers,
                    sys$TVP, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    # Convert to R list
    # if (command == "disturb"){
    #     sys$eta = output$eta
    #     sys$eps = output$eps
    # } else {
    #     sys$a = output$a
    #     sys$P = output$P
    #     sys$v = output$v
    #     sys$yFitV = output$yFitV
    #     sys$yFit = output$yFit
    # }
    # Re-building matrices to their original sizes
    n = length(sys$y) + sys$h
    # m = length(output$a) / n
    m = sum(sys$hidden$ns)
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        # aux = ts(matrix(NA, length(sys$y) + 1, 1), sY, frequency = fY)
        # sys$yFor = ts(output$yFor, end(aux), frequency = fY)
        # sys$yForV = ts(output$yForV, end(aux), frequency = fY)
        if (command == "disturb"){
            n = length(sys$y)
            mEta = length(output$eta) / n
            sys$eta = ts(t(matrix(output$eta, mEta, n)), sY, frequency = fY)
            sys$eps = ts(t(matrix(output$eps, 1, n)), sY, frequency = fY)
        } else {
            sys$a = ts(t(matrix(output$a, m, n)), sY, frequency = fY)
            # sys$aFor = ts(t(matrix(sys$aFor, m, sys$h)), end(aux), frequency = fY)
            sys$P = ts(t(matrix(output$P, m, n)), sY, frequency = fY)
            # sys$PFor = ts(t(matrix(sys$PFor, m, sys$h)), end(aux), frequency = fY)
            sys$yFit = ts(output$yFit, sY, frequency = fY)
            sys$yFitV = ts(output$yFitV, sY, frequency = fY)
            aux = ts(matrix(NA, length(sys$y) - length(output$v) + 1, 1), sY, frequency = fY)
            sys$v = ts(output$v, end(aux), frequency = fY)
        }
    } else {
        if (command == "disturb"){
            n = length(sys$y)
            sys$eta = t(matrix(output$eta, m, n))
            sys$eps = t(matrix(output$eps, 1, n))
        } else {
            sys$a = t(matrix(output$a, m, n))
            # sys$aFor = t(matrix(sys$aFor, m, sys$h))
            sys$P = t(matrix(output$P, m, n))
            # sys$PFor = t(matrix(sys$PFor, m, sys$h))
            sys$yFit = output$yFit
            sys$yFitV = output$yFitV
            sys$v = output$v
        }
    }
    if (command != "disturb"){
        names = strsplit(output$stateNames, "/")[[1]]
        nNames = length(names)
        if (is.vector(sys$a)){
            sys$a = matrix(sys$a, length(sys$a), 1)
            sys$P = matrix(sys$P, length(sys$P), 1)
        }
        if (ncol(sys$a) >= nNames){
            sys$a = sys$a[, 1 : nNames]
            sys$P = sys$P[, 1 : nNames]
            if (length(size(sys$a)) == 1){
                sys$a = matrix(sys$a, length(sys$a), 1)
                sys$P = matrix(sys$P, length(sys$P), 1)
            }
            colnames(sys$a) = strsplit(output$stateNames, "/")[[1]]
            colnames(sys$P) = strsplit(output$stateNames, "/")[[1]]
        }
    }
    # # States names
    # if (command != "disturb"){
    #     namesStates = c("Level")
    #     if (sys$hidden$ns[1] > 1){
    #         namesStates = c(namesStates, "Slope")
    #     }
    #     if (sys$hidden$ns[2] > 0){
    #         j = 1
    #         for (i in seq(1, sys$hidden$ns[2], 2)){
    #             namesStates = c(namesStates, paste0("Cycle", j))
    #             namesStates = c(namesStates, paste0("Cycle", j, "*"))
    #             j = j + 1
    #         }
    #     }
    #     if (sys$hidden$ns[3] > 0){
    #         nharm = ceiling(sys$hidden$ns[3] / 2)
    #         periods = tail(sys$periods, nharm)
    #         j = 1
    #         for (i in seq(1, sys$hidden$ns[3], 2)){
    #             namesStates = c(namesStates, paste0("Seasonal", j))
    #             if (periods[j] != 2){
    #                 namesStates = c(namesStates, paste0("Seasonal", j, "*"))
    #             }
    #             j = j + 1
    #         }
    #     }
    #     if (sys$hidden$ns[4] > 0){
    #         for (i in 1 : sys$hidden$ns[4]){
    #             if (i == 1){
    #                 namesStates = c(namesStates, paste0("Irregular", i))
    #             } else {
    #                 namesStates = c(namesStates, paste0("Irr*"))
    #             }
    #         }
    #     }
    #     colnames(sys$a) = namesStates
    #     colnames(sys$P) = namesStates
    # }
    return(sys)
}
#' @title MSOEfilter
#' @description Runs the Kalman Filter for MSOE models
#' Standard methods applicable to \code{MSOE} objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{MSOE} created with \code{MSOE}
#'
#' @return The same input object with the appropriate fields
#' filled in, in particular:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates (diagonal of covariance matrices)
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEvalidate}},
#'          \code{\link{MSOEsmooth}}, \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' m1 <- MSOE(log(AirPassengers))
#' m1 <- MSOEfilter(m1)
#' }
#' @rdname MSOEfilter
#' @export
MSOEfilter = function(sys){
    return(filter_(sys, "filter"))
}
#' @title MSOEsmooth
#' @description Runs the Fixed Interval Smoother for MSOE models
#' Standard methods applicable to \code{MSOE} objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{MSOE} created with \code{MSOE}
#'
#' @return The same input object with the appropriate fields
#' filled in, in particular:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates (diagonal of covariance matrices)
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}},
#'          \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' m1 <- MSOE(log(AirPassengers))
#' m1 <- MSOEsmooth(m1)
#' }
#' @rdname MSOEsmooth
#' @export
MSOEsmooth = function(sys){
    return(filter_(sys, "smooth"))
}
#' @title MSOEvalidate
#' @description Shows a table of estimation and diagnostics results for MSOE models.
#' Equivalent to print or summary.
#' The table shows information in four sections:
#' Firstly, information about the model estimated, the relevant
#' periods of the seasonal component included, and further information about
#' convergence.
#' Secondly, parameters with their names are provided, the asymptotic standard errors,
#' the ratio of the two, and the gradient at the optimum. One asterisk indicates
#' concentrated-out parameters and two asterisks signals parameters constrained during estimation.
#' Thirdly, information criteria and the value of the log-likelihood.
#' Finally, diagnostic statistics about innovations, namely, the Ljung-Box Q test of absense
#' of autocorrelation statistic for several lags, the Jarque-Bera gaussianity test, and a
#' standard ratio of variances test.
#'
#' @param sys an object of type \code{MSOE} created with \code{MSOE}
#' @param printScreen print to screen or just return output table
#'
#' @return The same input object with the appropriate fields
#' filled in, in particular:
#' \itemize{
#' \item table: Estimation and validation table
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEfilter}},
#'          \code{\link{MSOEsmooth}}, \code{\link{MSOEcomponents}},
#'
#'
#' @examples
#' \dontrun{
#' m1 <- MSOE(log(gdp))
#' m1 <- MSOEvalidate(m1)
#' }
#' @rdname MSOEvalidate
#' @export
MSOEvalidate = function(sys, printScreen = TRUE){
    if (is.ts(sys$y)){
        y = as.numeric(sys$y)
    } else {
        y = sys$y
    }
    if (is.ts(sys$u)){
        u = as.numeric(sys$u)
    } else {
        u = sys$u
    }
    if (any(is.na(sys$hidden$truePar))){
        if (printScreen){
            print(sys$table)
        }
        return(sys)
    }
    # Convert to R list
    #sys$periods = sys$hidden$periods0
    #sys$rhos = sys$hidden$rhos0
    rubbish = c(sys$hidden$d_t, sys$hidden$innVariance, sys$hidden$objFunValue, TRUE,
                sys$outlier, sys$arma, sys$iter, sys$hidden$seas, sys$lambda,
                sys$hidden$MSOE, sys$hidden$PTSnames)
    rubbish2 = cbind(sys$grad, sys$hidden$constPar, sys$hidden$typePar)
    rubbish3 = cbind(sys$hidden$ns, sys$hidden$nPar)
    output = MSOEc("validate", y, u, sys$model, sys$periods, sys$rhos,
                    sys$h, sys$tTest, sys$criterion, sys$hidden$truePar, rubbish2, rubbish, sys$verbose,
                    sys$stepwise, sys$hidden$estimOk, sys$p0, sys$v, sys$yFitV,
                    sys$hidden$nonStationaryTerms, rubbish3, sys$hidden$harmonics,
                    as.vector(sys$criteria), sys$hidden$cycleLimits,
                    cbind(sys$hidden$beta, sys$hidden$betaV), sys$hidden$typeOutliers,
                    sys$TVP, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    sys$table = output$table
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        aux = ts(matrix(NA, length(sys$y) - length(output$v) + 1, 1), sY, frequency = fY)
        sys$v = ts(output$v, end(aux), frequency = fY)
    } else {
        sys$v = output$v
    }
    if (printScreen){
        cat(output$table)
    }
    sys$covp = output$covp
    sys$p = as.vector(output$coef)
    # Parameter names from table
    nPar = length(sys$p)
    # parNames = rep("", nPar)
    # rowM = 2
    # hyphen = 1
    # i = 1
    # while (hyphen < 4){
    #     lineI = sys$table[rowM]
    #     if (substr(lineI, 1, 1) == "-"){
    #         hyphen = hyphen + 1
    #     }
    #     if (hyphen > 2 && substr(lineI, 1, 1) != "-"){
    #         parNames[i] = substr(lineI, 1, gregexpr(pattern =':', lineI))
    #         i = i + 1
    #     }
    #     rowM = rowM + 1
    # }
    # rownames(sys$covp) = parNames[1 : dim(sys$covp)[1]]
    # colnames(sys$covp) = parNames[1 : dim(sys$covp)[1]]
    # names(sys$p) = parNames[1 : nPar]
    rownames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
    colnames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
    names(sys$p) = output$parNames[1 : nPar]

    return(sys)
}
#' @title MSOEcomponents
#' @description Estimates unobserved components of MSOE models
#' Standard methods applicable to MSOEomp objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{MSOE} created with \code{MSOE} or \code{MSOEmodel}
#'
#' @return The same input object with the appropriate fields
#' filled in, in particular:
#' \itemize{
#' \item comp:  Estimated components in matrix form
#' \item compV: Estimated components variance in matrix form
#' }
#'
#' @template authors
#'
#' @seealso \code{\link{MSOE}}, \code{\link{MSOEmodel}}, \code{\link{MSOEvalidate}}, \code{\link{MSOEfilter}},
#'          \code{\link{MSOEsmooth}},
#'
#'
#' @examples
#' \dontrun{
#' m1 <- MSOE(log(AirPassengers))
#' m1 <- MSOEcomponents(m1)
#' }
#' @rdname MSOEcomponents
#' @export
MSOEcomponents= function(sys){
    if (is.ts(sys$y)){
        y = as.numeric(sys$y)
    } else {
        y = sys$y
    }
    if (is.ts(sys$u)){
        u = as.numeric(sys$u)
    } else {
        u = sys$u
    }
    rubbish = c(sys$hidden$d_t, sys$hidden$innVariance, sys$hidden$objFunValue, TRUE,
                sys$outlier, sys$arma, sys$iter, sys$hidden$seas, sys$lambda,
                sys$hidden$MSOE, sys$hidden$PTSnames)
    rubbish2 = cbind(sys$grad, sys$hidden$constPar, sys$hidden$typePar)
    rubbish3 = cbind(sys$hidden$ns, sys$hidden$nPar)
    output = MSOEc("components", y, u, sys$model, sys$periods, sys$rhos,
                    sys$h, sys$tTest, sys$criterion, sys$hidden$truePar, rubbish2, rubbish, sys$verbose,
                    sys$stepwise, sys$hidden$estimOk, sys$p0, sys$v, sys$yFitV,
                    sys$hidden$nonStationaryTerms, rubbish3, sys$hidden$harmonics,
                    as.vector(sys$criteria), sys$hidden$cycleLimits,
                    cbind(sys$hidden$beta, sys$hidden$betaV), sys$hidden$typeOutliers,
                    sys$TVP, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    # Convert to R list
    sys$comp = output$comp
    sys$compV = output$compV
    m = output$m  # + nCycles
    if (dim(u)[1] == 1 && dim(u)[2] == 2){
        k = 0
    } else {
        k = dim(u)[1]
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
    # namesComp = c("Level", "Slope", "Seasonal", "Irregular")
    # if (nCycles > 0){
    #     for (i in 1 : nCycles){
    #         namesComp = c(namesComp, paste0("Cycle", i))
    #     }
    # }
    # # Inputs names
    # if (k > 0){
    #     nOut = 0;
    #     if (sys$hidden$typeOutliers[1, 2] != -1){
    #         nOut = dim(sys$hidden$typeOutliers)[1]
    #     }
    #     nU = k - nOut
    #     if (nU > 0){
    #         for (i in 1 : nU){
    #             namesComp = c(namesComp, paste0("Exogenous", i))
    #         }
    #     }
    #     if (nOut > 0){
    #         for (i in 1 : nOut){
    #             namei = "AO"
    #             if (sys$hidden$typeOutliers[i, 1] == 1){
    #                 namei = "LS"
    #             } else if (sys$hidden$typeOutliers[i, 1] == 2){
    #                 namei = "SC"
    #             }
    #             namesComp = c(namesComp, paste0(namei, sys$hidden$typeOutliers[i, 2]))
    #         }
    #     }
    # }
    # # Eliminating components that are zero
    # n = dim(sys$comp)[1] - sys$h
    # ind = NULL
    # for (i in 1 : 3){
    #     if (max(sys$comp[1 : n, i], na.rm = TRUE) == 0){
    #         ind = c(ind, i)
    #     }
    # }
    # if (max(abs(sys$comp[1 : n, 4]), na.rm = TRUE) < 1e-12)
    #     ind = c(ind, 4)
    # if (length(ind) > 0){
    #     sys$comp = sys$comp[, -ind]
    #     sys$compV = sys$compV[, -ind]
    #     namesComp = namesComp[-ind]
    # }
    if (length(size(sys$comp)) == 1){
        if (is.ts(sys$y)){
            sys$comp = ts(matrix(sys$comp, n + sys$h, 1), start = start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
            sys$compV = ts(matrix(sys$compV, n + sys$h, 1), start = start(sys$y, frequency = frequency(sys$y)), frequency = frequency(sys$y))
        } else {
            sys$comp = matrix(sys$comp, n + sys$h, 1)
            sys$compV = matrix(sys$compV, n + sys$h, 1)
        }
    }
    # colnames(sys$comp) = namesComp
    # colnames(sys$compV) = namesComp
    # if (substr(output$compNames, nchar(output$compNames), nchar(output$compNames)) == "/")
    #     output$compNames = substr(output$compNames, 1, nchar(output$compNames) - 1)
    names = strsplit(output$compNames, "/")
    colnames(sys$comp) = names[[1]]
    colnames(sys$compV) = names[[1]]

    return(sys)
}
