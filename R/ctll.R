#' Continuous time Local Level model
#'
#' Function applies and estimates the continuous time local level model.
#'
#' Function applies a discretised version of the continuous time local level model.
#' This model can be applied to the demand that happens at irregular frequency. It
#' considers the zero values in the data as the absence of the value and only focuses
#' on the demand sizes of the data.
#'
#' @param y Vector or ts object, containing data needed to be forecasted.
#' @param type What type of variable to assume: stock or flow. Example of the stock
#' variable is a price of a product. Example of the flow variable is the income over
#' time.
#' @param h Length of forecasting horizon.
#' @param holdout If TRUE, then the holdout of the size h is taken from the data
#' (can be used for the model testing purposes).
#' @param silent Specifies, whether to provide the progress of the function
#' or not. If TRUE, then the function will print what it does and how much it
#' has already done.
#' @param log Whether to take logarithms of the demand sizes or not.
#' @param B The vector of initial parameters (variances).
#'
#' @template authors
#' @template keywords
#'
#' @return The object of the class ctll.
#'
#' @seealso \code{\link[UComp]{UC}}, \code{\link[smooth]{adam}}
#'
#' @examples
#' y <- rpois(100,1)
#' ctll(y)
#'
#' @rdname ctll
#' @export
ctll = function(y, u=NULL, type=c("stock", "flow"), log=FALSE,
                h=12, holdout=FALSE, silent=TRUE, B=c(0.1, 0.1)){
    # Copyright (C) 2024 - Inf  Diego Pedregal & Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    obsEq <- match.arg(type);

    #### This part allows us working with any class for y, not just ts
    #### Extract the class and indices to use the further in the code
    ### tsibble has its own index function, so shit happens because of it...
    if(inherits(y,"tbl_ts")){
        yIndex <- y[[1]];
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",yName,
                           ". I will refactor this."),call.=FALSE);
            yIndex <- yIndex[1] + c(1:length(y[[1]])) * diff(tail(yIndex,2));
        }
    }
    else{
        yIndex <- try(time(y),silent=TRUE);
        # If we cannot extract time, do something
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(y) && !is.null(dim(y))){
                yIndex <- as.POSIXct(rownames(y));
            }
            else if(is.data.frame(y)){
                yIndex <- c(1:nrow(y));
            }
            else{
                yIndex <- c(1:length(y));
            }
        }
    }
    yClasses <- class(y);
    # If this is just a numeric variable, use ts class
    if(all(yClasses=="integer") || all(yClasses=="numeric")){
        if(any(class(yIndex) %in% c("POSIXct","Date"))){
            yClasses <- "zoo";
        }
        else{
            yClasses <- "ts";
        }
    }
    yFrequency <- frequency(y);
    yStart <- yIndex[1];

    # Define obs, the number of observations of in-sample
    obsAll <- length(y) + (1 - holdout)*h;
    obsInSample <- length(y) - holdout*h;

    yInSample <- as.matrix(y[1:obsInSample]);
    if(holdout){
        yForecastStart <- yIndex[obsInSample+1];
        yHoldout <- y[-c(1:obsInSample)];
        yForecastIndex <- yIndex[-c(1:obsInSample)];
        yInSampleIndex <- yIndex[c(1:obsInSample)];
        yIndexAll <- yIndex;
    }
    else{
        yInSampleIndex <- yIndex;
        yIndexDiff <- diff(tail(yIndex,2));
        yForecastStart <- yIndex[obsInSample]+yIndexDiff;
        if(any(yClasses=="ts")){
            yForecastIndex <- yIndex[obsInSample]+as.numeric(yIndexDiff)*c(1:max(h,1));
        }
        else{
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1));
        }
        yHoldout <- NULL;
        yIndexAll <- c(yIndex,yForecastIndex);
    }

    otLogical <- yInSample!=0;

    out =  list(y = yInSample,
                u = u,
                h = h,
                obsEq = obsEq,
                p0 = B,
                verbose = !silent,
                logTransform = log,
                # outputs
                yFor = NULL,
                yForV = NULL,
                comp = NULL,
                compV = NULL,
                table = "",
                p = NULL)
    # smooth is here to get support for default plots etc.
    m = structure(out, class=c("ctll","smooth"))
    # Re-doing u
    if (!is.null(m$u)){
        if (is.vector(m$u)){
            u = matrix(m$u, 1, length(m$u))
        } else {
            nu = dim(m$u)
            u = as.numeric(m$u);
            u = matrix(u, nu[1], nu[2])
        }
    }
    # Running C++ code
    output = INTLEVELc("e", yInSample, u, h, obsEq, !silent, B, log)
    # Preparing outputs
    if (length(output) == 1){   # ERROR!!
        stop()
    } else {
        # m$p = output$p
        lu = size(m$u)[1]
        if (lu > 0)
            m$h = lu - obsInSample

        # Create proper ts objects of fitted, forecast, residuals etc
        if(any(yClasses=="ts")){
            m$y <- ts(yInSample, start=yStart, frequency=yFrequency)
            m$fitted <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency)
            m$residuals <- ts(output$comp[1:obsInSample,1], start=yStart, frequency=yFrequency)
            if(h>0){
                m$forecast <- ts(output$yFor, start=yForecastStart, frequency=yFrequency)
                m$variance <- ts(output$yForV, start=yForecastStart, frequency=yFrequency)
                if(holdout){
                    m$holdout <- ts(yHoldout, start=yForecastStart, frequency=yFrequency)
                }
            }
        }
        else{
            m$y <- zoo(yInSample, order.by=yInSampleIndex)
            m$fitted <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex)
            m$residuals <- zoo(output$comp[,1], order.by=yInSampleIndex)
            if(h>0){
                m$forecast <- zoo(output$yFor, order.by=yForecastIndex)
                m$variance <- zoo(output$yForV, order.by=yForecastIndex)
                if(holdout){
                    m$holdout <- zoo(yHoldout, order.by=yForecastIndex)
                }
            }
        }
        m$fitted[] <- output$comp[1:obsInSample,2]
        #### Temporary solution with interpolated values ####
        if(any(is.nan(m$fitted))){
            m$fitted[is.nan(m$fitted)] <- NA
            m$fitted[] <- approx(m$fitted, xout=c(1:obsInSample), rule=2)$y
        }
        m$residuals[is.nan(m$residuals)] <- NA

#         # Create
#         if(is.ts(m$yInSample) && m$h > 0){
#             fake = ts(c(m$y, NA), start = start(m$y), frequency = frequency(m$y))
#             m$yFor = ts(output$yFor, start = end(fake), frequency = frequency(m$y))
#             m$yForV = ts(output$yForV, start = end(fake), frequency = frequency(m$y))
#             m$comp = ts(output$comp, start = start(m$y), frequency = frequency(m$y))
#             m$compV = ts(output$compV, start = start(m$y), frequency = frequency(m$y))
#         } else if (m$h > 0) {
#             m$yFor = output$yFor
#             m$yForV = output$yForV
#             m$ySimul = output$ySimul
#             m$comp = output$comp
#             m$compV = output$compV
#         }

        # colnames(m$comp) = strsplit(output$compNames, split = "/")[[1]]
        m$table = output$table
        m$timeElapsed <- Sys.time()-startTime

        m$states <- m$comp[,3,drop=FALSE]
        # Variance of the residuals
        m$s2 <- sum(m$residuals^2, na.rm=TRUE)/(nobs(m))
        # Estimated parameters
        m$B <- setNames(as.vector(output$p), c("Var(eta)"))
        m$model <- "Continuous Time Local Level Model"
        m$log <- log

        if(log){
            m$logLik <- sum(dlnorm(y[otLogical], m$fitted[otLogical], sqrt(m$s2), log=TRUE))
            m$fitted[] <- exp(m$fitted)
            m$forecast[] <- exp(m$forecast)
        }
        else{
            m$logLik <- sum(dnorm(y[otLogical], m$fitted[otLogical], sqrt(m$s2), log=TRUE))
        }

        m$comp <- NULL
        m$p <- NULL
        m$yFor <- NULL
        m$yForV <- NULL

        if (!silent){
            cat("Done!\n")
            # print(m)
            plot(m, 7)
        }

        return(m)
    }
}

#' @export
print.ctll <- function(x, digits=4, ...){
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds\n");
    cat(x$table);

    cat("\nSample size:", nobs(x));
    cat("\nNumber of estimated parameters:", nparam(x));
    cat("\nNumber of degrees of freedom:", nobs(x)-nparam(x));

    ICs <- c(AIC(x),AICc(x),BIC(x),BICc(x));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat("\nInformation criteria:\n");
    print(round(ICs,digits));
}

#' @export
coef.ctll <- function(object, ...){
    return(object$B);
}

#' @export
nparam.ctll <- function(object, ...){
    return(length(coef(object)));
}

#' @export
residuals.ctll <- function(object, ...){
    return(object$residuals);
}

#' @rdname ctll
#' @param object The estimated ctll object.
#' @param interval The type of the interval to construct. Currently, only
#' the prediction interval is supported.
#' @param level The confidence level, the value lying between 0 and 1.
#' @param side Whether to produce a prediction interval (\code{"both"})
#' or produce an upper/lower quantile only.
#' @param cumulative Logical, specifying whether to return the cumulative
#' point forecasts and quantiles.
#' @param ... Other parameters (not yet used).
#' @importFrom generics forecast
#' @export
forecast.ctll <- function(object, h=10, interval=c("prediction","none"),
                          level=0.95, side=c("both", "upper", "lower"),
                          cumulative=FALSE, ...){
    yInSample <- actuals(object)

    output = INTLEVELc(yInSample, object$u, h, object$obsEq, FALSE, object$B, object$log)
}
