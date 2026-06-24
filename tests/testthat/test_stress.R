# Stress / invariant tests across a broad grid of pts() configurations.
# These assert structural invariants (not machine-specific golden values),
# so they are portable and serve as a lasting regression net — complementing
# the numeric golden-master harness in tools/characterize.R.

## ---- fixtures -------------------------------------------------------------
set.seed(42)
stress_data <- list(
    air     = AirPassengers,
    air_log = log(AirPassengers),
    bjsales = BJsales,
    trendy  = ts(pmax(1, round(50 + 2 * (1:60) + 15 * sin(2 * pi * (1:60) / 12)
                               + rnorm(60, 0, 5))), frequency = 12)
)
stress_specs <- c("ZZZ", "1NN", "1LN", "1LT", "1DT", "1GT", "1ND",
                  "0LT", "0.5LT", "2LT", "1ZN", "ZNN")

## ---- the whole grid fits and yields a coherent object ---------------------
test_that("pts fits across the spec x data grid with coherent invariants", {
    for (dn in names(stress_data)) {
        y <- stress_data[[dn]]
        for (spec in stress_specs) {
            info <- sprintf("data=%s spec=%s", dn, spec)
            m <- pts(y, model = spec, h = 6, holdout = TRUE)
            expect_s3_class(m, "pts")
            # in-sample lengths
            n_in <- length(y) - 6
            expect_equal(length(fitted(m)),    n_in, info = info)
            expect_equal(length(residuals(m)), n_in, info = info)
            expect_equal(nobs(m),              n_in, info = info)
            # finiteness
            expect_true(all(is.finite(fitted(m))),    info = info)
            expect_true(all(is.finite(residuals(m))), info = info)
            expect_true(is.finite(as.numeric(logLik(m))), info = info)
            expect_true(is.finite(sigma(m)) && sigma(m) > 0, info = info)
            # forecast
            expect_equal(length(m$forecast), 6L, info = info)
            expect_true(all(is.finite(m$forecast)), info = info)
            # comp / states shape
            expect_true(all(c("Error", "Fit") %in% colnames(m$comp)), info = info)
            expect_equal(nrow(m$states), n_in + 1L, info = info)
            # nParam coherence: counts at least the free coefficients
            expect_true(nparam(m) >= length(coef(m)) - 1L, info = info)
            expect_true(nparam(m) >= 1L, info = info)
        }
    }
})

## ---- information criteria orderings --------------------------------------
test_that("information criteria are finite and self-consistent", {
    for (dn in names(stress_data)) {
        y <- stress_data[[dn]]
        m <- pts(y, model = "ZZZ", h = 0)
        ic <- c(AIC(m), BIC(m), greybox::AICc(m), greybox::BICc(m))
        expect_true(all(is.finite(ic)), info = dn)
        # small-sample corrected ICs are >= their base versions
        expect_gte(greybox::AICc(m), AIC(m) - 1e-8)
        expect_gte(greybox::BICc(m), BIC(m) - 1e-8)
        # AIC = -2 logLik + 2k
        k <- nparam(m)
        expect_equal(AIC(m), -2 * as.numeric(logLik(m)) + 2 * k, tolerance = 1e-6,
                     info = dn)
    }
})

## ---- prediction intervals are ordered and nest by level -------------------
test_that("prediction intervals are ordered and monotone in level", {
    m <- pts(AirPassengers, model = "ZZZ", h = 12)
    fc <- forecast(m, h = 12, interval = "prediction", level = c(0.8, 0.95))
    lo <- as.matrix(fc$lower); up <- as.matrix(fc$upper)
    # lower <= mean <= upper, columnwise
    expect_true(all(lo[, 1] <= fc$mean + 1e-6))
    expect_true(all(up[, 1] >= fc$mean - 1e-6))
    # 95% band contains the 80% band
    expect_true(all(lo[, 2] <= lo[, 1] + 1e-6))
    expect_true(all(up[, 2] >= up[, 1] - 1e-6))
    # positive series with lambda>=0: lower bound stays non-negative
    expect_true(all(lo >= -1e-6))
})

test_that("confidence intervals are narrower than prediction intervals", {
    m <- pts(AirPassengers, model = "ZZZ", h = 12)
    fp <- forecast(m, h = 12, interval = "prediction", level = 0.95)
    fcid <- forecast(m, h = 12, interval = "confidence", level = 0.95)
    width_p <- as.numeric(fp$upper) - as.numeric(fp$lower)
    width_c <- as.numeric(fcid$upper) - as.numeric(fcid$lower)
    expect_true(all(width_c <= width_p + 1e-6))
})

## ---- forecast side / cumulative options ----------------------------------
test_that("one-sided and cumulative forecasts behave", {
    m <- pts(AirPassengers, model = "1LT", h = 12)
    fu <- forecast(m, h = 12, interval = "prediction", level = 0.95, side = "upper")
    fl <- forecast(m, h = 12, interval = "prediction", level = 0.95, side = "lower")
    expect_equal(length(fu$mean), 12L)
    expect_equal(length(fl$mean), 12L)
    fcum <- forecast(m, h = 12, interval = "prediction", level = 0.95,
                     cumulative = TRUE)
    expect_equal(length(fcum$mean), 1L)
    expect_true(is.finite(fcum$mean))
})

## ---- ARMA on the irregular ------------------------------------------------
test_that("ARMA irregular specs fit and report orders", {
    for (ord in list(list(ar = 1, ma = 0), list(ar = 0, ma = 1),
                     list(ar = 1, ma = 1))) {
        m <- pts(AirPassengers, model = "1LT", h = 6, orders = ord)
        o <- orders(m)
        expect_equal(o$ar, ord$ar)
        expect_equal(o$ma, ord$ma)
        expect_true(all(is.finite(residuals(m))))
    }
})

test_that("orders$select runs and stays within caps", {
    m <- pts(AirPassengers, model = "1LT", h = 6, ic = "AICc",
             orders = list(ar = 2, ma = 2, select = TRUE))
    o <- orders(m)
    expect_true(o$ar <= 2L && o$ma <= 2L)
    # the select flag round-trips on the stored $orders slot
    expect_true(isTRUE(m$orders$select))
})

## ---- diagnostics methods --------------------------------------------------
test_that("diagnostic methods return finite, correctly-sized output", {
    m <- pts(AirPassengers, model = "ZZZ", h = 0)
    n <- nobs(m)
    expect_equal(length(rstandard(m)), n)
    expect_equal(length(rstudent(m)),  n)
    expect_equal(length(pointLik(m)),  n)
    expect_true(all(is.finite(rstandard(m))))
    expect_true(all(is.finite(pointLik(m))))
    # rstandard is approximately unit-variance
    expect_lt(abs(sd(rstandard(m)) - 1), 0.5)
})

## ---- simulate -------------------------------------------------------------
test_that("simulate produces an (nobs x nsim) original-scale matrix", {
    # simulate.pts replays the in-sample period from the initial state, so
    # the path length is nobs (not the forecast horizon).
    m <- pts(AirPassengers, model = "ZZZ", h = 0)
    s <- simulate(m, nsim = 50, seed = 7)
    sm <- as.matrix(s$data)
    expect_equal(nrow(sm), nobs(m))
    expect_equal(ncol(sm), 50L)
    expect_true(all(is.finite(sm)))
    # reproducible under a fixed seed
    s2 <- simulate(m, nsim = 50, seed = 7)
    expect_equal(sm, as.matrix(s2$data))
})

## ---- accuracy on holdout --------------------------------------------------
test_that("accuracy returns finite error measures on the holdout", {
    m <- pts(AirPassengers, model = "ZZZ", h = 12, holdout = TRUE)
    acc <- accuracy(m)
    expect_true(is.numeric(acc))
    expect_true(any(is.finite(acc)))
})

## ---- summary / confint / update ------------------------------------------
test_that("summary, confint and update work end to end", {
    m <- pts(AirPassengers, model = "1LT", h = 6)
    sm <- summary(m)
    expect_s3_class(sm, "summary.pts")
    expect_true(is.matrix(sm$coefficients) || is.data.frame(sm$coefficients))
    ci <- suppressWarnings(confint(m, level = 0.9))
    expect_equal(nrow(as.matrix(ci)), length(coef(m)))
    m2 <- update(m, h = 10)
    expect_s3_class(m2, "pts")
    expect_equal(length(m2$forecast), 10L)
})

## ---- Box-Cox back-transform sanity ---------------------------------------
test_that("fitted/forecast are positive when lambda in (0,1] for positive data", {
    for (lam in c("0LT", "0.5LT", "1LT")) {
        m <- pts(AirPassengers, model = lam, h = 12)
        expect_true(all(fitted(m) > 0), info = lam)
        expect_true(all(m$forecast > 0), info = lam)
    }
})

## ---- missing values: lambda screen must still run -------------------------
test_that("auto-lambda screen runs through missing values (not forced to 1)", {
    # AirPassengers is multiplicative -> the screen picks lambda ~ 0 (log).
    # A few NAs must not collapse the screen back to lambda = 1: msdecompose
    # imputes the gaps and the block stats use na.rm.
    yc <- AirPassengers
    yn <- yc; yn[41:43] <- NA
    # Exercise the screen explicitly (it is opt-in now that the default lambda
    # estimator is the engine's joint likelihood).
    mc <- pts(yc, model = "ZZZ", h = 0, lambda_estim = "decomp-guerrero")
    mn <- pts(yn, model = "ZZZ", h = 0, lambda_estim = "decomp-guerrero")
    expect_lt(mn$lambda, 0.5)                          # not collapsed to 1
    expect_lt(abs(mn$lambda - mc$lambda), 1e-3)        # matches the gap-free fit
    # residuals are finite except at the (3) missing observations
    expect_gte(sum(is.finite(residuals(mn))), nobs(mn) - 3L)
})
