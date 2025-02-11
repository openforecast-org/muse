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
#' @param u DIEGO NEEDS TO EXPLAIN WHAT THIS IS
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
#' @importFrom zoo zoo
#' @rdname ctll
#' @export
ctll = function(y, u=NULL, type=c("stock", "flow"), log=FALSE,
                h=12, holdout=FALSE, silent=TRUE, B=c(0.1, 0.1)){
    # Copyright (C) 2024 - Inf  Diego Pedregal & Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    obsEq <- match.arg(type);

    cl <- match.call();

    #### This part allows us working with any class for y, not just ts
    #### Extract the class and indices to use the further in the code
    ### tsibble has its own index function, so shit happens because of it...
    if(inherits(y,"tbl_ts")){
        yIndex <- y[[1]];
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",
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

    # Re-doing u
    if (!is.null(u)){
        if (is.vector(u)){
            u = matrix(u, 1, length(u))
        } else {
            nu = dim(u)
            u = as.numeric(u);
            u = matrix(u, nu[1], nu[2])
        }
    }
    # Running C++ code
    output = INTLEVELc("e", yInSample, u, h, obsEq, !silent, B, log)

    # cat("This are the new fields:\n")
    # print(cbind(output$yForAgg, output$yForVAgg))

    # Preparing outputs
    if (length(output) == 1){   # ERROR!!
        stop()
    } else {

        # smooth is here to get support for default plots etc.
        m <- structure(list(y = yInSample), class=c("ctll","smooth"))

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
        # mu is the location of the distribution
        m$mu <- m$fitted
        m$residuals[is.nan(m$residuals)] <- NA

        m$type <- obsEq
        m$B0 <- B
        # Estimated parameters
        # exp(2*p) is what Diego said is the variance of eta
        m$B <- setNames(as.vector(exp(2*output$p)), c("Var(eta)"))
        m$silent <- silent
        m$log <- log

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

        m$states <- output$comp[,3,drop=FALSE]
        # Variance of the residuals
        m$scale <- sum(m$residuals^2, na.rm=TRUE)/(nobs(m))
        m$model <- "Continuous Time Local Level Model"
        m$log <- log

        if(log){
            m$logLik <- sum(dlnorm(yInSample[otLogical], m$fitted[otLogical], sqrt(m$scale), log=TRUE), na.rm=TRUE)
            #### Fitted and Forecast here correspond to the mean of the log Normal distribution
            m$fitted[] <- exp(m$fitted + m$variance[1]/2)
            m$forecast[] <- exp(as.vector(m$forecast) + as.vector(m$variance)/2)
        }
        else{
            m$logLik <- sum(dnorm(yInSample[otLogical], m$fitted[otLogical], sqrt(m$scale), log=TRUE), na.rm=TRUE)
        }

        m$comp <- NULL
        m$p <- NULL
        m$yFor <- NULL
        m$yForV <- NULL
        m$u <- u
        m$call <- cl

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
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat(paste0("\nModel estimated using ",tail(all.vars(x$call[[1]]),1),
               "() function: ",x$model));
    cat(x$table);

    cat("\nSample size:", nobs(x));
    cat("\nNumber of estimated parameters:", nparam(x));
    cat("\nNumber of degrees of freedom:", nobs(x)-nparam(x));

    ICs <- c(AIC(x),AICc(x),BIC(x),BICc(x));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat("\nInformation criteria:\n");
    print(round(ICs,digits));
}

#' @importFrom stats coef
#' @export
coef.ctll <- function(object, ...){
    return(object$B);
}

#' @importFrom stats nobs
#' @export
nparam.ctll <- function(object, ...){
    return(length(actuals(object)));
}

#' @importFrom greybox nparam
#' @export
nparam.ctll <- function(object, ...){
    return(length(coef(object)));
}

#' @importFrom stats residuals
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
    side <- match.arg(side);
    interval <- match.arg(interval);

    yInSample <- actuals(object);
    obsInSample <- nobs(object);

    B <- object$B
    # if(object$log){
        # B <- exp(2*object$B)
    # }

    yIndex <- time(yInSample);
    yClasses <- class(yInSample);
    # Create indices for the future
    if(any(yClasses=="ts")){
        # ts structure
        yForecastStart <- time(yInSample)[obsInSample]+deltat(yInSample);
        yFrequency <- frequency(yInSample);
        yForecastIndex <- yIndex[obsInSample]+as.numeric(diff(tail(yIndex,2)))*c(1:h);
    }
    else{
        # zoo
        yIndex <- time(yInSample);
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        hFinal <- h;
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yVariance <- yMean <- yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        if(cumulative){
            yVariance <- yMean <- yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex[1]);
            yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex[1]);
        }
        else{
            yVariance <- yMean <- yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
            yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
        }
    }

    output <- INTLEVELc("f", as.matrix(yInSample), object$u, h,
                        object$type, FALSE, B, object$log)

    if(cumulative){
        yMean[] <- yForecast[] <- output$yForAgg[h]
        yVariance[] <- output$yForVAgg[h]
    }
    else{
        yMean[] <- yForecast[] <- output$yFor
        yVariance[] <- output$yForV
    }

    if(interval=="prediction"){
        if(side=="upper"){
            yLower[] <- rep(-Inf, hFinal)
            yUpper[] <- qnorm(level, mean=yMean, sd=sqrt(yVariance))
        }
        else if(side=="both"){
            yLower[] <- qnorm((1-level)/2, mean=yMean, sd=sqrt(yVariance))
            yUpper[] <- qnorm((1+level)/2, mean=yMean, sd=sqrt(yVariance))
        }
        else{
            yLower[] <- qnorm(1-level, mean=yMean, sd=sqrt(yVariance))
            yUpper[] <- rep(Inf, hFinal)
        }
    }
    else{
        yLower <- yUpper <- NULL;
    }

    if(object$log){
        yMean[] <- exp(yMean + yVariance/2);
        yLower[] <- exp(yLower)
        yUpper[] <- exp(yUpper)
    }

    # Names for quantiles
    if(interval!="none"){
        colnames(yLower) <- switch(side,
                                   "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                                   "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels),
                                   "upper"=paste0("Upper bound (",level*100,"%)"));
    }

    return(structure(list(model=object, mean=yMean, mu=yForecast, variance=yVariance,
                          lower=yLower, upper=yUpper, level=level, h=h,
                          side=side, interval=interval, cumulative=cumulative),
                     class="ctll.forecast"));
}

#' @export
print.ctll.forecast <- function(x, ...){
    if(x$interval!="none"){
        returnedValue <- switch(x$side,
                                "both"=cbind(x$mean,x$lower,x$upper),
                                "lower"=cbind(x$mean,x$lower),
                                "upper"=cbind(x$mean,x$upper));
        colnames(returnedValue) <- switch(x$side,
                                          "both"=c("Point forecast",colnames(x$lower),colnames(x$upper)),
                                          "lower"=c("Point forecast",colnames(x$lower)),
                                          "upper"=c("Point forecast",colnames(x$upper)))
    }
    else{
        returnedValue <- x$mean;
    }
    print(returnedValue);
}

#' @method plot ctll.forecast
#' @importFrom greybox actuals
#' @export
plot.ctll.forecast <- function(x, ...){
    yClasses <- class(actuals(x$model));

    ellipsis <- list(...);

    if(is.null(ellipsis$legend)){
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
    }

    if(is.null(ellipsis$main)){
        ellipsis$main <- paste0("Forecast from the ",x$model$model);
    }

    if(!is.null(x$model$holdout)){
        yHoldout <- x$model$holdout;
        if(any(yClasses=="ts")){
            ellipsis$actuals <- ts(c(actuals(x$model),yHoldout),
                                   start=start(actuals(x$model)),
                                   frequency=frequency(actuals(x$model)));
        }
        else{
            ellipsis$actuals <- zoo(c(as.vector(actuals(x$model)),as.vector(yHoldout)),
                                    order.by=c(time(actuals(x$model)),time(yHoldout)));
        }
    }
    else{
        ellipsis$actuals <- actuals(x$model);
    }

    ellipsis$forecast <- x$mean;
    ellipsis$lower <- x$lower;
    ellipsis$upper <- x$upper;
    ellipsis$fitted <- fitted(x$model);
    ellipsis$level <- x$level;

    if(x$cumulative){
        # Create cumulative actuals and fitted values
        obsInSample <- nobs(x$model);
        obsNew <- floor(obsInSample/x$h);
        yCum <- yFittedCum <- vector("numeric", obsNew);
        ellipsis$actuals[is.na(ellipsis$actuals)] <- 0;
        for(i in 1:obsNew){
            yCum[i] <- sum(ellipsis$actuals[obsInSample-(x$h:1)+1-(obsNew-i)*x$h], na.rm=TRUE);
            yFittedCum[i] <- sum(ellipsis$fitted[obsInSample-(x$h:1)+1-(obsNew-i)*x$h], na.rm=TRUE);
        }

        # Get deltat to see where to place forecast
        if(!is.null(x$model$holdout)){
            yDeltat <- 0;
            yCum <- c(yCum, sum(ellipsis$actuals[obsInSample+(1:x$h)]), na.rm=TRUE);
        }
        else{
            yDeltat <- deltat(ellipsis$actuals);
        }
        yFreqCum <- frequency(ellipsis$actuals)/x$h;

        # Form new ts objects based on the cumulative values
        # We ignore zoo here because who cares...
        ellipsis$actuals <- ts(yCum,
                               start=start(ellipsis$actuals),
                               frequency=yFreqCum);
        ellipsis$fitted <- ts(yFittedCum,
                              start=start(ellipsis$actuals),
                              frequency=yFreqCum);
        ellipsis$forecast <- ts(ellipsis$forecast,
                                start=end(ellipsis$actuals)+yDeltat,
                                frequency=frequency(ellipsis$forecast));
        ellipsis$lower <- ts(ellipsis$lower,
                             start=end(ellipsis$actuals)+yDeltat,
                             frequency=frequency(ellipsis$forecast));
        ellipsis$upper <- ts(ellipsis$upper,
                             start=end(ellipsis$actuals)+yDeltat,
                             frequency=frequency(ellipsis$forecast));
        ellipsis$main <- paste0("Cumulative ", ellipsis$main);
        ellipsis$vline <- FALSE;
    }

    do.call(graphmaker, ellipsis);

    # A fix for weird frequencies for the cumulative forecasts
    if(x$cumulative){
        points(ellipsis$actuals);
        abline(v=tail(time(ellipsis$fitted),1),col="red",lwd=2);
    }
}
