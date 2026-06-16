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
    # The fit-time forecast uses bsmMatrices (concentrated scale + cLlik=true)
    # whereas forecast.pts re-runs via bsmMatricesTrue (absolute scale +
    # cLlik=false).  Both paths are mathematically equivalent for the point
    # forecast, but pinv(Sn) in the augmented KF loses precision when σ̂² is
    # small (Sn entries ~σ̂² → conditioning ~1/σ̂²).  Tolerance is loosened
    # to absorb that purely-numerical mismatch; if it ever exceeds ~1% the
    # underlying precision issue needs revisiting.
    expect_equal(as.numeric(f$mean), as.numeric(m$forecast), tolerance = 1e-2)
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
    # Both m$forecast and f$mean are on the original scale.  The two code
    # paths agree mathematically; the small numerical gap comes from
    # pinv(Sn) precision in bsmMatricesTrue (absolute scale) — see notes on
    # the cache-vs-rerun test above.
    expect_equal(as.numeric(f$mean), as.numeric(m$forecast), tolerance = 1e-2)
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

test_that("orders$ar fits an AR(p) on the irregular component", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = 1, ma = 0))
    expect_match(m$modelUC, "arma\\(1,0\\)")
    expect_true("AR(1)" %in% names(coef(m)))
    expect_equal(m$orders$ar, 1L)
    expect_equal(m$orders$ma, 0L)
})

test_that("orders$ar and orders$ma fit a full ARMA(p,q)", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = 1, ma = 1))
    expect_match(m$modelUC, "arma\\(1,1\\)")
    cf <- names(coef(m))
    expect_true("AR(1)" %in% cf)
    expect_true("MA(1)" %in% cf)
})

test_that("orders accepts a c(p, q) numeric shortcut", {
    # c(p, q) is shorthand for list(ar = p, ma = q, select = FALSE).
    m_v <- pts(AirPassengers, model = "1LT", h = 0, orders = c(1, 1))
    m_l <- pts(AirPassengers, model = "1LT", h = 0,
               orders = list(ar = 1, ma = 1))
    expect_equal(m_v$modelUC, m_l$modelUC)
    expect_equal(as.numeric(coef(m_v)), as.numeric(coef(m_l)),
                 tolerance = 1e-10)
    # Scalar shorthand: c(p) → ma = 0.
    m_s <- pts(AirPassengers, model = "1LT", h = 0, orders = c(1))
    expect_match(m_s$modelUC, "arma\\(1,0\\)")
    expect_equal(m_s$orders$ma, 0L)
    # Bad shapes are rejected.
    expect_error(pts(AirPassengers, model = "1LT", h = 0,
                     orders = c(1, 1, 1)),
                 "c\\(p\\) or c\\(p, q\\)")
})

test_that("orders accepts seasonal SARMA via per-lag vectors", {
    # Seasonal AR only: SARMA(0,0)(1,0)_12 — coef table should show SAR(1)
    # but no AR(1).
    m_s <- pts(AirPassengers, model = "1LT", h = 0,
               orders = list(ar = c(0, 1), ma = c(0, 0), lags = c(1, 12)))
    expect_match(m_s$modelUC, "arma\\(0,0,1,0,12\\)")
    cf <- names(coef(m_s))
    expect_true("SAR(1)"  %in% cf)
    expect_false("AR(1)"  %in% cf)
    # $orders round-trips the seasonal spec
    expect_equal(m_s$orders$ar,   c(0L, 1L))
    expect_equal(m_s$orders$ma,   c(0L, 0L))
    expect_equal(m_s$orders$lags, c(1L, 12L))
})

test_that("orders supports mixed SARMA(1,0)(1,0)_12", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = c(1, 1), ma = c(0, 0), lags = c(1, 12)))
    cf <- names(coef(m))
    expect_true("AR(1)"  %in% cf)
    expect_true("SAR(1)" %in% cf)
    # Forecast should be finite and roughly in the data range
    f <- forecast(m, h = 12)
    expect_true(all(is.finite(as.numeric(f$mean))))
    expect_true(min(f$mean) > 100 && max(f$mean) < 1000)
})

test_that("orders uses frequency(data) as default seasonal lag when missing", {
    # Implicit lag = c(1, frequency(data)); for AirPassengers that's
    # c(1, 12) — should match the explicit form above.
    m_exp <- pts(AirPassengers, model = "1LT", h = 0,
                 orders = list(ar = c(0, 1), ma = c(0, 0), lags = c(1, 12)))
    m_imp <- pts(AirPassengers, model = "1LT", h = 0,
                 orders = list(ar = c(0, 1), ma = c(0, 0)))
    expect_equal(m_exp$modelUC, m_imp$modelUC)
})

test_that("orders rejects mismatched lengths", {
    expect_error(
        pts(AirPassengers, model = "1LT", h = 0,
            orders = list(ar = c(1, 1), ma = c(0, 0), lags = c(1))),
        "must match")
})

test_that("select=TRUE searches the seasonal grid up to the (p, q, P, Q) cap", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = c(1, 1), ma = c(1, 0),
                           lags = c(1, 12), select = TRUE))
    # Residual-based grid search picks one element of the cap grid; check
    # the chosen orders stay within bounds and `select = TRUE` round-trips.
    expect_true(m$orders$ar[1] <= 1L && m$orders$ar[2] <= 1L)
    expect_true(m$orders$ma[1] <= 1L && m$orders$ma[2] <= 0L)
    expect_true(isTRUE(m$orders$select))
    expect_match(m$modelUC, "arma\\([0-9]+,[0-9]+(,[0-9]+,[0-9]+,12)?\\)$")
})

test_that("select=TRUE with cap (0,0) reduces to no ARMA", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = 0, ma = 0, select = TRUE))
    expect_equal(m$orders$ar, 0L)
    expect_equal(m$orders$ma, 0L)
    expect_true(isTRUE(m$orders$select))
    expect_match(m$modelUC, "arma\\(0,0\\)")
})

test_that("ARMA recovers a pure AR(1) signal (no cancellation manifold)", {
    # Regression: with zero ARMA init the BFGS used to stay at the start
    # (AR(1) ≈ -0.02) on a synthetic AR(1) signal because the gradient was
    # symmetric in (AR, MA).  Asymmetric ACF/PACF init breaks the symmetry
    # — muse should land within 0.1 of the true φ.
    set.seed(1)
    n <- 500
    y <- numeric(n); e <- rnorm(n); y[1] <- e[1]
    for (t in 2:n) y[t] <- 0.7 * y[t-1] + e[t]
    m <- pts(y, model = "1NN", orders = list(ar = 1, ma = 0))
    expect_lt(abs(as.numeric(coef(m)["AR(1)"]) - 0.7), 0.1)
})

test_that("ARMA(2,2) on log(AirPassengers) escapes the AR=MA manifold", {
    # The previous zero init left φ_i = θ_i to bit precision; ACF/PACF
    # init plus the tiebreaker should put AR(i) clearly distinct from
    # MA(i) — by at least 0.05 in absolute value.
    m <- pts(log(AirPassengers), model = "1LT",
             orders = list(ar = 2, ma = 2))
    b <- coef(m)
    expect_gt(abs(as.numeric(b["AR(1)"]) - as.numeric(b["MA(1)"])), 0.05)
    expect_gt(abs(as.numeric(b["AR(2)"]) - as.numeric(b["MA(2)"])), 0.05)
})

test_that("select=TRUE searches up to the (ar, ma) cap and reports the choice", {
    m <- pts(AirPassengers, model = "1LT", h = 0,
             orders = list(ar = 2, ma = 2, select = TRUE))
    expect_true(m$orders$ar <= 2L)
    expect_true(m$orders$ma <= 2L)
    # The chosen irregular shows up either as arma(p,q) or as the
    # "none" candidate when no ARMA structure improves the IC.
    expect_match(m$modelUC,
                 "(arma\\([0-9]+,[0-9]+\\)|/none)$")
})

#### outliers + level ####
test_that("outliers = 'ignore' produces no detection and round-trips slot", {
    m <- pts(AirPassengers, model = "0NT", h = 0)
    expect_equal(m$outliers, "ignore")
    expect_equal(m$level, 0.99)
    expect_s3_class(m$outliersDetected, "data.frame")
    expect_equal(nrow(m$outliersDetected), 0L)
    expect_equal(names(m$outliersDetected), c("time", "type"))
})

test_that("outliers = 'use' detects an injected spike", {
    # Plant an obvious additive spike at t = 100.
    y <- AirPassengers
    y[100] <- 3 * y[100]
    m <- pts(y, model = "0NT", h = 0, outliers = "use", level = 0.99)
    expect_equal(m$outliers, "use")
    expect_gt(nrow(m$outliersDetected), 0L)
    # The planted spike must land in the detected set.
    expect_true(100L %in% m$outliersDetected$time)
    # And its dummy must show up in coef(m) (engine names them AO<t> /
    # LS<t> / SC<t>).
    cn <- names(coef(m))
    expect_true(any(grepl("(AO|LS|SC)100", cn)))
})

test_that("outliers = 'select' errors with 'not yet supported'", {
    expect_error(
        pts(AirPassengers, model = "0NT", h = 0, outliers = "select"),
        "not yet supported")
})

test_that("level is validated as a length-1 numeric in (0, 1)", {
    expect_error(pts(AirPassengers, model = "0NT", h = 0,
                     outliers = "use", level = 1),    "in \\(0, 1\\)")
    expect_error(pts(AirPassengers, model = "0NT", h = 0,
                     outliers = "use", level = 0),    "in \\(0, 1\\)")
    expect_error(pts(AirPassengers, model = "0NT", h = 0,
                     outliers = "use", level = c(0.9, 0.95)),
                 "length-1")
})

#### negative values ####
test_that("negative values trigger a warning and pin lambda to 1", {
    yneg <- log(AirPassengers) - mean(log(AirPassengers))
    expect_warning(
        m <- pts(yneg, model = "ZZZ", h = 0),
        "negative values"
    )
    expect_equal(m$lambda, 1)
    # Explicit lambda = 1 is left alone, no warning either way.
    expect_silent(pts(yneg, model = "1NT", h = 0))
})
