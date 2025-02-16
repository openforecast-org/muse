#' @title size
#' @description size of vectors or matrices
#'
#' @param y matrix, array or vector
#'
#' @author Diego J. Pedregal
#'
#' @rdname size
#' @export
size = function(y){
    out = dim(y)
    if (is.null(out))
        out = length(y)
    return(out)
}
#' @title print.PTS
#' @description Prints a PTS object
#'
#' @details See help of \code{PTS}.
#'
#' @param x Object of class \dQuote{PTS}.
#' @param ... Additional inputs to handle the way to print output.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{PTS}}, \code{\link{PTSmodel}}, \code{\link{PTSvalidate}},
#'          \code{\link{PTScomponents}}, \code{\link{PTSestim}}
#'
#' @examples
#' # m1 <- PTSmodel(log(AirPassengers))
#' # print(m1)
#' @rdname print
#' @export
print.PTS = function(x, ...){
    if (length(x$table) == 0){
        x = PTSvalidate(x)
    } else {
        cat(x$table)
    }
}
#' @title summary.PTS
#' @description Prints an PTS object on screen
#'
#' @param object Object of class \dQuote{PTS}.
#' @param ... Additional inputs to function.
#'
#' @details See help of \code{PTS}.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{PTS}}, \code{\link{PTSmodel}}, \code{\link{PTSvalidate}},
#'          \code{\link{PTScomponents}}, \code{\link{PTSestim}}
#'
#' @examples
#' # m1 <- PTSmodel(log(AirPassengers))
#' # summary(m1)
#' @rdname summary
#' @export
summary.PTS = function(object, ...){
    print(object)
}
#' @title plot.PTS
#' @description Plot components of PTS object
#'
#' @details See help of \code{PTS}.
#'
#' @param x Object of class \dQuote{PTS}.
#' @param ... Additional inputs to function.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{PTS}}, \code{\link{PTSmodel}}, \code{\link{PTSvalidate}},
#'          \code{\link{PTScomponents}}, \code{\link{PTSestim}}
#'
#' @examples
#' # m1 <- PTS(log(AirPassengers))
#' # plot(m1)
#' @rdname plot
#' @export
plot.PTS = function(x, ...){
    if (length(x$comp) < 2){
        x = PTScomponents(x)
    }
    if (is.ts(x$comp)){
        plot(x$comp, main = "Time Series Decomposition")
    } else {
        plot(ts(x$comp, frequency = x$s),
             main = "Time Series Decomposition")
    }
}
#' @title fitted.PTS
#' @description Fitted output values of PTS object
#'
#' @details See help of \code{PTS}.
#'
#' @param object Object of class \dQuote{PTS}.
#' @param ... Additional inputs to function.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{PTS}}, \code{\link{PTSmodel}}, \code{\link{PTSvalidate}},
#'          \code{\link{PTScomponents}}, \code{\link{PTSestim}}
#'
#' @examples
#' # m1 <- PTSmodel(log(AirPassengers))
#' # fitted(m1)
#' @rdname fitted
#' @export
fitted.PTS = function(object, ...){
    if (length(object$comp) < 2){
        object = PTScomponents(object)
    }
    return(object$com[, 2])
}
#' @title residuals.PTS
#' @description Residuals of PTS object
#'
#' @details See help of \code{PTS}.
#'
#' @param object Object of class \dQuote{PTS}.
#' @param ... Additional inputs to function.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{PTS}}, \code{\link{PTSmodel}}, \code{\link{PTSvalidate}},
#'          \code{\link{PTScomponents}}, \code{\link{PTSestim}}
#'
#' @examples
#' # m1 <- PTSmodel(log(AirPassengers))
#' # residuals(m1)
#' @rdname residuals
#' @export
residuals.PTS = function(object, ...){
    if (length(object$comp) < 2){
        object = PTScomponents(object)
    }
    return(object$com[, 1])
}


