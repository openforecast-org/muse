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

# uc_to_pts: invert pts_to_uc, given a fully-resolved UC string and the
# estimated lambda.
uc_to_pts <- function(modelUC, lambda){
    modelUC <- sub("/none/", "/", modelUC, fixed = TRUE)
    sl <- gregexpr("/", modelUC)[[1]]
    trend    <- substr(modelUC, 1,         sl[1] - 1)
    seasonal <- substr(modelUC, sl[1] + 1, sl[2] - 1)
    model <- as.character(round(lambda, 2))
    if      (trend == "rw")  model <- paste0(model, "N")
    else if (trend == "srw") model <- paste0(model, "D")
    else if (trend == "llt") model <- paste0(model, "L")
    else if (trend == "td")  model <- paste0(model, "G")
    if      (seasonal == "none")   model <- paste0(model, "N")
    else if (seasonal == "equal")  model <- paste0(model, "T")
    else if (seasonal == "linear") model <- paste0(model, "D")
    model
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
