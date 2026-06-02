# A seasonal series so that PTS exercises the seasonal path (PTS$comp building
# blows up on series with neither slope nor seasonal components).
y <- log(AirPassengers)

#### PTS: object structure ####
test_that("PTS returns a populated PTS object", {
    m <- PTS(y, model = "0NT", h = 12)
    expect_s3_class(m, "PTS")
    expect_equal(length(m$yFor), 12)
    expect_equal(length(m$yForV), 12)
    expect_true(is.ts(m$yFor))
    expect_true(is.ts(m$yForV))
    expect_equal(frequency(m$yFor), frequency(y))
    expect_false(any(is.na(m$yFor)))
    expect_false(any(is.na(m$yForV)))
    expect_true(all(m$yForV > 0))
    expect_equal(m$lambda, 0)
    expect_equal(m$model, "0NT")
    expect_true(is.matrix(m$comp))
    expect_equal(nrow(m$comp), length(y) + 12)
    expect_true(all(c("Error", "Fit") %in% colnames(m$comp)))
    expect_true("Seasonal" %in% colnames(m$comp))
    expect_equal(length(m$v), length(y))
    expect_true(any(nchar(m$table) > 0))
    expect_s3_class(m$modelUC, "MSOE")
})

test_that("PTS honours the forecast horizon", {
    m12 <- PTS(y, model = "0NT", h = 12)
    m24 <- PTS(y, model = "0NT", h = 24)
    expect_equal(length(m12$yFor), 12)
    expect_equal(length(m24$yFor), 24)
    expect_equal(length(m12$yForV), 12)
    expect_equal(length(m24$yForV), 24)
})

test_that("PTS preserves the requested Box-Cox lambda", {
    expect_equal(PTS(y, model = "0NT", h = 12)$lambda, 0)
    expect_equal(PTS(y, model = "1NT", h = 12)$lambda, 1)
    expect_equal(PTS(y, model = "0.5NT", h = 12)$lambda, 0.5)
})

test_that("PTS preserves the requested trend / seasonal letters", {
    expect_equal(substr(PTS(y, model = "0NT", h = 12)$model, 2, 3), "NT")
    expect_equal(substr(PTS(y, model = "0LT", h = 12)$model, 2, 3), "LT")
    expect_equal(substr(PTS(y, model = "0DT", h = 12)$model, 2, 3), "DT")
    expect_equal(substr(PTS(y, model = "0ND", h = 12)$model, 2, 3), "ND")
})

test_that("PTS accepts aic, bic and aicc criteria", {
    expect_silent(m_aic  <- PTS(y, model = "0NT", h = 12, criterion = "aic"))
    expect_silent(m_bic  <- PTS(y, model = "0NT", h = 12, criterion = "bic"))
    expect_silent(m_aicc <- PTS(y, model = "0NT", h = 12, criterion = "aicc"))
    expect_equal(m_aic$criterion,  "aic")
    expect_equal(m_bic$criterion,  "bic")
    expect_equal(m_aicc$criterion, "aicc")
    expect_equal(length(m_aic$yFor),  12)
    expect_equal(length(m_bic$yFor),  12)
    expect_equal(length(m_aicc$yFor), 12)
})

test_that("PTS Fit column is the row-sum of the structural components", {
    m <- PTS(y, model = "0NT", h = 12)
    structural <- m$comp[, -c(1, 2), drop = FALSE]
    expect_equal(as.numeric(m$comp[, "Fit"]),
                 as.numeric(rowSums(structural)),
                 tolerance = 1e-10)
})

test_that("PTS armaIdent path runs and stores arma orders", {
    m <- PTS(y, model = "0NT", h = 12, armaIdent = TRUE)
    expect_s3_class(m, "PTS")
    expect_length(m$armaOrders, 2)
    expect_true(all(m$armaOrders >= 0))
})

#### PTSforecast ####
test_that("PTSforecast returns a PTS object with the same forecast length as PTS", {
    mf <- PTSforecast(y, model = "0NT", h = 12)
    expect_s3_class(mf, "PTS")
    expect_equal(length(mf$yFor), 12)
    expect_equal(length(mf$yForV), 12)
    expect_true(is.ts(mf$yFor))
    expect_false(any(is.na(mf$yFor)))
    expect_true(all(mf$yForV > 0))
    expect_s3_class(mf$modelUC, "MSOE")
})

test_that("PTSforecast and PTS agree on forecasts for the same model", {
    m  <- PTS(y, model = "0NT", h = 12)
    mf <- PTSforecast(y, model = "0NT", h = 12)
    expect_equal(as.numeric(mf$yFor),  as.numeric(m$yFor),  tolerance = 1e-4)
    expect_equal(as.numeric(mf$yForV), as.numeric(m$yForV), tolerance = 1e-4)
    expect_equal(mf$lambda, m$lambda)
})

test_that("PTSforecast honours the forecast horizon", {
    expect_equal(length(PTSforecast(y, model = "0NT", h =  6)$yFor),  6)
    expect_equal(length(PTSforecast(y, model = "0NT", h = 18)$yFor), 18)
})

test_that("PTSforecast forces lambda = 1 for non-seasonal series", {
    set.seed(7)
    yns <- ts(cumsum(rnorm(60)))
    mf <- PTSforecast(yns, model = "1NN", h = 5)
    expect_equal(mf$lambda, 1)
    expect_equal(length(mf$yFor), 5)
})

test_that("PTSforecast accepts a numeric vector when s is supplied", {
    yv <- as.numeric(y)
    mf <- PTSforecast(yv, model = "0NT", s = 12, h = 12)
    expect_s3_class(mf, "PTS")
    expect_equal(length(mf$yFor), 12)
})
