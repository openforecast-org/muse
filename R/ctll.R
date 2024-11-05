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
#' @param silent Specifies, whether to provide the progress of the function
#' or not. If TRUE, then the function will print what it does and how much it
#' has already done.
#' @param log Whether to take logarithms of the demand sizes or not.
#' @param B The vector of initial parameters (variances).
#'
#' @template authors
#' @template keywords
#'
#' @return
#'
#' @seealso \code{\link[UComp]{UC}}, \code{\link[smooth]{adam}}
#'
#' @examples
#' y <- rpois(100,1)
#' ctll(y)
#'
#' @rdname ctll
#' @export
ctll = function(y, u=NULL, type=c("stock", "flow"), log=TRUE,
                h=12, silent=TRUE, B=c(0.1, 0.1)){
    # Copyright (C) 2024 - Inf  Diego Pedregal & Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    obsEq <- match.arg(type);

    if(!is.ts(y)){
        y = as.ts(y)
    }
    out =  list(y = y,
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
    m = structure(out, class = "ctll")
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
    output = INTLEVELc(as.numeric(m$y), u, h, obsEq, !silent, B, log)
    # Preparing outputs
    if (length(output) == 1){   # ERROR!!
        stop()
    } else {
        m$p = output$p
        lu = size(m$u)[1]
        if (lu > 0)
            m$h = lu - length(m$y)
        if (is.ts(m$y) && m$h > 0){
            fake = ts(c(m$y, NA), start = start(m$y), frequency = frequency(m$y))
            m$yFor = ts(output$yFor, start = end(fake), frequency = frequency(m$y))
            m$yForV = ts(output$yForV, start = end(fake), frequency = frequency(m$y))
            m$comp = ts(output$comp, start = start(m$y), frequency = frequency(m$y))
            m$compV = ts(output$compV, start = start(m$y), frequency = frequency(m$y))
        } else if (m$h > 0) {
            m$yFor = output$yFor
            m$yForV = output$yForV
            m$ySimul = output$ySimul
            m$comp = output$comp
            m$compV = output$compV
        }
        colnames(m$comp) = strsplit(output$compNames, split = "/")[[1]]
        m$table = output$table;
        if (!silent){
            cat(m$table)
        }
        m$timeElapsed <- Sys.time()-startTime;

        return(m)
    }
}

#' @export
print.ctll <- function(x, ...){
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds\n");
    cat(x$table)
}
