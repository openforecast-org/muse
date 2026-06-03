# update.pts -- re-run pts() with substituted arguments, mirroring
# stats::update.default but pinned to pts so dispatch stays clean.

#' @export
update.pts <- function(object, ...){
    cl <- object$call
    if (is.null(cl))
        stop("pts object has no recorded call; cannot update.", call. = FALSE)
    extras <- list(...)
    for (nm in names(extras))
        cl[[nm]] <- extras[[nm]]
    eval(cl, envir = parent.frame())
}
