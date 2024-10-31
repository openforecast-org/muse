#' @title funName
#' @description 
#'
#' @details 
#'
#' @param y 
#' 
#' @author 
#' 
#' @return 
#' 
#' @seealso 
#'          
#' @examples
#' \dontrun{}
#' @rdname funName
#' @export
funName = function(y, u = NULL, h = 12, obsEq = "stock", verbose = FALSE, 
                   p0 = c(0.1, 0.1), logTransform=FALSE){
    y = as.ts(y)
    out =  list(y = y,
                u = u,
                h = h,
                obsEq = obsEq,
                p0 = p0,
                verbose = verbose,
                logTransform = logTransform,
                # outputs
                yFor = NULL,
                yForV = NULL,
                comp = NULL,
                compV = NULL,
                table = "",
                p = NULL)
    m = structure(out, class = "intLevel")
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
    output = INTLEVELc(as.numeric(m$y), u, h, obsEq, verbose, p0, logTransform)
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
        if (verbose)
            cat(m$table)
        return(m)
    }
}
    