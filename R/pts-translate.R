# Internal translators between PTS 3-letter spec and the UC model string
# that the C++ engine speaks. Not exported.

# pts_to_uc: turn a PTS spec ("0NT", "ZZZ", "0.5LD") into a UC string and
# a Box-Cox lambda. armaOrders sets the irregular ARMA(p,q) on the UC side.
pts_to_uc <- function(model, armaOrders = c(0, 0)){
    out <- list(modelU = "", lambda = 1.0)
    n   <- nchar(model)
    # ARMA(p,q) irregular component
    modelU <- paste0("/arma(", armaOrders[1], ",", armaOrders[2], ")")
    # Seasonal
    aux <- tolower(substr(model, n, n))
    if      (aux == "z") modelU <- paste0("/?",     modelU)
    else if (aux == "n") modelU <- paste0("/none",  modelU)
    else if (aux == "d") modelU <- paste0("/linear",modelU)
    else if (aux == "t") modelU <- paste0("/equal", modelU)
    else stop("Invalid seasonal letter in PTS spec: '", aux, "'", call. = FALSE)
    # Trend
    aux <- tolower(substr(model, n - 1, n - 1))
    if      (aux == "z") modelU <- paste0("?/none",  modelU)
    else if (aux == "n") modelU <- paste0("rw/none", modelU)
    else if (aux == "l") modelU <- paste0("llt/none",modelU)
    else if (aux == "d") modelU <- paste0("srw/none",modelU)
    else if (aux == "g") modelU <- paste0("td/none", modelU)
    else stop("Invalid trend letter in PTS spec: '", aux, "'", call. = FALSE)
    out$modelU <- modelU
    # Power (Box-Cox lambda): either numeric or 'Z' for estimate
    aux <- tolower(substr(model, 1, n - 2))
    num <- suppressWarnings(as.numeric(aux))
    if (!is.na(num))      out$lambda <- num
    else if (aux == "z")  out$lambda <- 9999.9   # C++ sentinel for "estimate"
    else stop("Invalid power letter in PTS spec: '", aux, "'", call. = FALSE)
    out
}

# uc_to_pts: format a fully-resolved UC string + lambda as
#   PTS(<lambda>,<trend letter>,<seasonal letter>)
# e.g. PTS(0,N,T), PTS(-0.29,D,T), PTS(1,L,N).  Unknown trend / seasonal
# tokens collapse to an empty field rather than silently dropping a slot.
uc_to_pts <- function(modelUC, lambda){
    modelUC <- sub("/none/", "/", modelUC, fixed = TRUE)
    sl <- gregexpr("/", modelUC)[[1]]
    trend    <- substr(modelUC, 1,         sl[1] - 1)
    seasonal <- substr(modelUC, sl[1] + 1, sl[2] - 1)
    trendLetter <- switch(trend,
                          "rw"  = "N",
                          "srw" = "D",
                          "llt" = "L",
                          "td"  = "G",
                          "?"   = "Z",
                          "")
    seasonalLetter <- switch(seasonal,
                             "none"   = "N",
                             "equal"  = "T",
                             "linear" = "D",
                             "?"      = "Z",
                             "")
    sprintf("PTS(%s,%s,%s)",
            as.character(round(lambda, 2)),
            trendLetter,
            seasonalLetter)
}

# .pts_orders_to_uc: turn adam-style orders = list(ar, ma, select) into the
# (ar, ma) tuple + select flag that the UC engine consumes.  PTS has no
# differencing (engine has no `i` flow) and ARMA only sits on the irregular
# component, so we validate i == 0 (when supplied) and reject negatives.
.pts_orders_to_uc <- function(orders){
    if (is.null(orders)) orders <- list()
    ar <- if (is.null(orders$ar)) 0L else as.integer(orders$ar[1])
    ma <- if (is.null(orders$ma)) 0L else as.integer(orders$ma[1])
    sel <- isTRUE(orders$select)
    if (!is.null(orders$i) && any(orders$i != 0))
        stop("`orders$i` (differencing) is not supported by PTS — ",
             "PTS has no I component.", call. = FALSE)
    if (ar < 0 || ma < 0)
        stop("`orders$ar` and `orders$ma` must be non-negative integers.",
             call. = FALSE)
    list(ar = ar, ma = ma, select = sel)
}

# .pts_ic_to_engine: map adam-style ic (AICc / AIC / BIC / BICc) to the
# engine's lowercase criterion.  BICc collapses to BIC at the engine level;
# the small-sample correction is computed in R via BICc.pts.
.pts_ic_to_engine <- function(ic){
    ic <- match.arg(ic, c("AICc", "AIC", "BIC", "BICc"))
    switch(ic,
           "AICc" = "aicc",
           "AIC"  = "aic",
           "BIC"  = "bic",
           "BICc" = "bic")
}

# uc_to_arma: pull (p, q) out of "arma(p,q)" embedded in a UC string.
uc_to_arma <- function(model){
    p1 <- regexpr("arma", model, fixed = TRUE)[1]
    p1 <- p1 + 5
    p2 <- regexpr(",", model, fixed = TRUE)[1] - 1
    p3 <- regexpr(")", model, fixed = TRUE)[1] - 1
    ar <- suppressWarnings(as.numeric(substr(model, p1,     p2)))
    ma <- suppressWarnings(as.numeric(substr(model, p2 + 2, p3)))
    if (is.na(ar)) ar <- 0
    if (is.na(ma)) ma <- 0
    c(ar, ma)
}
