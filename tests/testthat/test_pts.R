# Tests for the new pts() / forecast.pts() API.

y <- log(AirPassengers)

#### pts: object structure ####
test_that("pts returns a populated 'pts' object", {
    m <- pts(y, model = "0NT", h = 12)
    expect_s3_class(m, "pts")
    expect_s3_class(m, "smooth")
    expect_equal(m$model, "PTS(0,N,T)")
    expect_equal(m$lambda, 0)
    expect_equal(m$lags, frequency(y))
    expect_true(is.matrix(m$comp))
    expect_true(all(c("Error", "Fit") %in% colnames(m$comp)))
    # fitted() / residuals() are in-sample only (length nobs), matching
    # standard R conventions; the full (n + h_fit) trajectory lives on $comp.
    expect_equal(length(fitted(m)),   length(y))
    expect_equal(length(residuals(m)), length(y))
    expect_true(length(m$forecast)  == 12)
    expect_true(is.ts(m$forecast))
    expect_false(any(is.na(m$forecast)))
})

test_that("pts honours h = 0 (forecast slot is ts(NA) placeholder)", {
    m <- pts(y, model = "0NT", h = 0)
    # adam.R:572 stores ts(NA, ...) when no forecast was requested;
    # pts mirrors that so $forecast is always a defined ts object.
    expect_equal(length(m$forecast), 1L)
    expect_true(is.na(m$forecast))
    expect_true(is.ts(m$forecast))
    expect_true(is.matrix(m$comp))
})

test_that("pts holdout splits y", {
    m <- pts(y, model = "0NT", h = 12, holdout = TRUE)
    expect_equal(length(m$data), length(y) - 12)
    expect_equal(length(m$holdout), 12)
    expect_equal(nobs(m), length(y) - 12)
})

test_that("pts ic argument is honoured (adam-style options)", {
    expect_silent(m_aicc <- pts(y, model = "0NT", h = 12, ic = "AICc"))
    expect_silent(m_aic  <- pts(y, model = "0NT", h = 12, ic = "AIC"))
    expect_silent(m_bic  <- pts(y, model = "0NT", h = 12, ic = "BIC"))
    expect_silent(m_bicc <- pts(y, model = "0NT", h = 12, ic = "BICc"))
    expect_equal(m_aic$ic,  "AIC")
    expect_equal(m_aicc$ic, "AICc")
    expect_equal(length(m_aic$forecast),  12)
    expect_equal(length(m_bicc$forecast), 12)
})

#### auto.pts ####
test_that("auto.pts resolves the model and returns 'pts'", {
    m <- auto.pts(y, h = 6)
    expect_s3_class(m, "pts")
    expect_true(nchar(m$model) >= 3)
    expect_match(m$modelUC, "/")
})

#### accessors ####
test_that("S3 accessors return sensible values", {
    m <- pts(y, model = "0NT", h = 12)
    expect_equal(length(coef(m)), m$nParam)
    expect_equal(dim(vcov(m)), c(m$nParam, m$nParam))
    expect_equal(nobs(m), length(y))
    ll <- logLik(m)
    expect_s3_class(ll, "logLik")
    expect_equal(attr(ll, "df"), m$nParam)
    expect_false(is.na(as.numeric(ll)))
    expect_false(is.na(AIC(m)))
    expect_false(is.na(BIC(m)))
})

test_that("predict.pts returns fitted values", {
    m <- pts(y, model = "0NT", h = 12)
    expect_equal(predict(m), fitted(m))
})

#### forecast.pts ####
test_that("forecast.pts returns a forecast object", {
    m <- pts(y, model = "0NT", h = 12)
    f <- forecast(m, h = 24)
    expect_s3_class(f, "pts.forecast")
    expect_equal(length(f$mean),  24)
    expect_equal(length(f$lower), 24)
    expect_equal(length(f$upper), 24)
    expect_true(all(f$lower <= f$mean))
    expect_true(all(f$mean  <= f$upper))
    expect_true(is.ts(f$mean))
    expect_equal(frequency(f$mean), frequency(y))
})

test_that("forecast.pts matches the forecast cached at fit time", {
    m <- pts(y, model = "0NT", h = 12)
    f <- forecast(m, h = 12)
    expect_equal(as.numeric(f$mean), as.numeric(m$forecast), tolerance = 1e-8)
})

test_that("forecast.pts honours the level argument", {
    m  <- pts(y, model = "0NT", h = 12)
    f9 <- forecast(m, h = 12, level = 0.90)
    f5 <- forecast(m, h = 12, level = 0.50)
    expect_lt(max(f5$upper - f5$lower),
              max(f9$upper - f9$lower))
})

test_that("forecast.pts rejects bad h", {
    m <- pts(y, model = "0NT", h = 12)
    expect_error(forecast(m, h = 0),  "positive integer")
    expect_error(forecast(m, h = -1), "positive integer")
})

#### S3 dispatch via base generics ####
test_that("base generics dispatch correctly", {
    m <- pts(y, model = "0NT", h = 12)
    # fitted / residuals are exposed in-sample only; truncate the cached
    # full trajectories to match.
    n <- length(m$data)
    expect_equal(as.numeric(fitted(m)),    as.numeric(m$fitted)[seq_len(n)])
    expect_equal(as.numeric(residuals(m)), as.numeric(m$residuals)[seq_len(n)])
    expect_identical(coef(m),              m$B)
})

#### Box-Cox back-transform ####
test_that("fitted() and yFor are on the original scale (lambda = 1 -> identity)", {
    m <- pts(y, model = "1NT", h = 12)
    expect_equal(m$lambda, 1)
    # With lambda = 1 the engine's BoxCox is the identity, so fitted ==
    # comp[, "Fit"] (in-sample portion).
    n <- length(m$data)
    expect_equal(as.numeric(fitted(m)),
                 as.numeric(m$comp[, "Fit"])[seq_len(n)])
})

test_that("fitted() and yFor are back-transformed when lambda = 0 (log/exp)", {
    m <- pts(y, model = "0NT", h = 12)
    expect_equal(m$lambda, 0)
    # comp[, "Fit"] is the engine's level on the BC scale (= log(y_scale)).
    # fitted() should be its exp(.) inverse on the original scale, in-sample.
    n <- length(m$data)
    expect_equal(as.numeric(fitted(m)),
                 exp(as.numeric(m$comp[, "Fit"]))[seq_len(n)])
    # m$forecast is the cached forecast on the original scale.
    expect_true(all(m$forecast > 0))
    # Sanity: y is log(AirPassengers) ~ 4.7-6.4. yFor stays in that band,
    # not in the BC band (which would be ~1.5 = log of 4.7).
    expect_true(min(m$forecast) > 4)
    expect_true(max(m$forecast) < 8)
})

test_that("fitted() matches a handcrafted inverse Box-Cox when 0 < lambda < 1", {
    m <- pts(y, model = "0.5NT", h = 6)
    expect_equal(m$lambda, 0.5)
    bc  <- as.numeric(m$comp[, "Fit"])
    n   <- length(m$data)
    # invBoxCox(z, 0.5) = (0.5*z + 1)^2, in-sample only
    expect_equal(as.numeric(fitted(m)), ((0.5 * bc + 1) ^ 2)[seq_len(n)])
})

test_that("forecast() returns asymmetric intervals on the original scale when lambda < 1", {
    m <- pts(y, model = "0NT", h = 12)
    f <- forecast(m, h = 12, level = 0.95)
    # On the BC scale intervals are symmetric around the point forecast;
    # after invBoxCox the original-scale intervals lose that symmetry.
    upper_arm <- as.numeric(f$upper - f$mean)
    lower_arm <- as.numeric(f$mean  - f$lower)
    expect_true(all(upper_arm > 0))
    expect_true(all(lower_arm > 0))
    # The upper arm should be visibly longer (Box-Cox with lambda=0 = log
    # makes the exp() inverse stretch the upper tail more).
    expect_true(mean(upper_arm) > mean(lower_arm))
})

test_that("forecast() round-trips with the cached yFor on the original scale", {
    m <- pts(y, model = "0NT", h = 12)
    f <- forecast(m, h = 12)
    # Both m$forecast and f$mean are on the original scale.  Their match (to
    # numerical noise) demonstrates the back-transform is identical in
    # both the `all` and `forecastOnly` code paths.
    expect_equal(as.numeric(f$mean), as.numeric(m$forecast), tolerance = 1e-10)
})

#### data + formula + regressors ####
test_that("pts() accepts a matrix; column 1 is the response", {
    xreg <- seq_along(y) / length(y)
    M    <- cbind(y = as.numeric(y), x = xreg)
    m    <- pts(M, model = "0NT", h = 0)
    expect_s3_class(m, "pts")
    expect_equal(m$responseName, "y")
    # Engine sees the regressor: $u is k x n with k = 1
    expect_equal(nrow(m$u), 1L)
    expect_equal(ncol(m$u), length(y))
    # A "Beta(1)" coefficient row is present in the C++ par-names list
    expect_true(any(grepl("^Beta", names(coef(m)))))
})

test_that("pts() accepts a formula on a data.frame", {
    df  <- data.frame(y = as.numeric(y),
                      x = seq_along(y) / length(y))
    m_M <- pts(cbind(df$y, df$x), model = "0NT", h = 0)
    m_F <- pts(df, formula = y ~ x, model = "0NT", h = 0)
    # The two paths feed the engine identical inputs, so the fits agree
    expect_equal(as.numeric(coef(m_F)), as.numeric(coef(m_M)),
                 tolerance = 1e-10)
    expect_equal(m_F$responseName, "y")
})

test_that("forecast.pts(newdata = …) supplies future xreg values", {
    df    <- data.frame(y = as.numeric(y),
                        x = seq_along(y) / length(y))
    m_xy  <- pts(df, formula = y ~ x, model = "0NT", h = 0)
    nd    <- data.frame(x = (length(y) + 1:12) / length(y))
    f     <- forecast(m_xy, h = 12, newdata = nd)
    expect_equal(length(f$mean), 12L)
    expect_true(all(is.finite(as.numeric(f$mean))))
})

test_that("forecast.pts errors helpfully when newdata is missing / wrong shape", {
    df    <- data.frame(y = as.numeric(y),
                        x = seq_along(y) / length(y))
    m_xy  <- pts(df, formula = y ~ x, model = "0NT", h = 0)
    expect_error(forecast(m_xy, h = 12), "newdata")
    expect_error(forecast(m_xy, h = 12, newdata = data.frame(x = 1:3)),
                 "must have at least")
})

test_that("orders argument replaces the old armaIdent flag", {
    m_sel <- pts(y, model = "0NT", h = 0,
                 orders = list(ar = 0, ma = 0, select = TRUE))
    expect_s3_class(m_sel, "pts")
    expect_true(m_sel$orders$select)
})
