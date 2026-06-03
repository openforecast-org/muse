# Smoke + invariant tests for the smooth/adam-style method surface we
# ported in Phase 7. One block per method family.

y <- log(AirPassengers)
m <- pts(y, model = "0NT", h = 12, holdout = TRUE)

#### accessors ####
test_that("sigma.pts returns sd of BC-scale innovations", {
    expect_equal(sigma(m), sd(residuals(m), na.rm = TRUE), tolerance = 1e-12)
})

test_that("nparam.pts matches the stored nParam slot", {
    expect_equal(nparam(m), m$nParam)
})

test_that("modelType.pts returns the PTS(lambda,T,S) code", {
    expect_equal(modelType(m), m$model)
    expect_match(modelType(m), "^PTS\\(.+,.+,.+\\)$")
})

test_that("modelName.pts spells out trend / seasonal", {
    nm <- modelName(m)
    expect_match(nm, "Trend=")
    expect_match(nm, "Seasonal=")
})

test_that("lags.pts returns the seasonal lag", {
    expect_equal(lags(m), frequency(y))
})

test_that("orders.pts returns ARMA orders for arma(0,0)", {
    o <- orders(m)
    expect_equal(o, list(ar = 0L, i = 0L, ma = 0L))
})

test_that("errorType.pts reports additive on the BC scale", {
    expect_equal(errorType(m), "A")
})

test_that("actuals.pts returns the y series", {
    expect_equal(as.numeric(actuals(m)), as.numeric(m$y))
})

test_that("AICc / BICc match the standard formulas", {
    n <- nobs(m); k <- nparam(m); ll <- as.numeric(logLik(m))
    expect_equal(AICc(m), 2*k - 2*ll + 2*k*(k+1)/(n - k - 1), tolerance = 1e-12)
    expect_equal(BICc(m), -2*ll + (k * log(n) * n) / (n - k - 1),  tolerance = 1e-12)
})

test_that("extractSigma / extractScale equal sigma", {
    expect_equal(extractSigma(m), sigma(m))
    expect_equal(extractScale(m), sigma(m))
})

#### diagnostics ####
test_that("rstandard.pts equals residuals / sigma", {
    r <- as.numeric(rstandard(m))
    e <- as.numeric(residuals(m)) / sigma(m)
    expect_equal(r, e, tolerance = 1e-12)
})

test_that("rstudent.pts is finite where rstandard is finite", {
    rs <- rstudent(m); rt <- rstandard(m)
    finite <- !is.na(as.numeric(rs))
    expect_equal(sum(finite), sum(!is.na(as.numeric(rt))))
})

test_that("pointLik.pts returns Gaussian density per residual", {
    pl <- pointLik(m, log = TRUE)
    expect_equal(length(pl), length(residuals(m)))
    # Manually compute Gaussian log-density at one finite point.
    e <- as.numeric(residuals(m)); s <- sigma(m)
    i <- which(!is.na(e))[1]
    expect_equal(as.numeric(pl)[i], dnorm(e[i], 0, s, log = TRUE),
                 tolerance = 1e-12)
})

test_that("outlierdummy returns bounds + id vector", {
    od <- outlierdummy(m, level = 0.99)
    expect_equal(length(od$statistic), 2L)
    expect_equal(od$statistic[1], -od$statistic[2])
    expect_true(all(od$id > 0L))
})

#### confint ####
test_that("confint.pts produces a 2-column matrix with the right dims", {
    suppressWarnings(ci <- confint(m, level = 0.9))
    expect_equal(dim(ci), c(nparam(m), 2L))
    finite <- complete.cases(ci)
    if (any(finite)){
        est <- coef(m); ses <- sqrt(diag(vcov(m)))
        i <- which(finite)[1]
        z <- qnorm(0.95)
        expect_equal(unname(ci[i, 1]), unname(est[i] - z * ses[i]),
                     tolerance = 1e-10)
        expect_equal(unname(ci[i, 2]), unname(est[i] + z * ses[i]),
                     tolerance = 1e-10)
    }
})

#### summary ####
test_that("summary.pts coefficient table has the adam columns", {
    s <- summary(m)
    expect_s3_class(s, "summary.pts")
    expect_equal(colnames(s$coefficients),
                 c("Estimate", "Std. Error", "t value", "Pr(>|t|)", "Lower", "Upper"))
    df <- as.data.frame(s)
    expect_true("Parameter" %in% names(df))
    expect_equal(nrow(df), nparam(m))
})

test_that("summary.pts numerical invariants", {
    s <- summary(m)
    cmat <- s$coefficients
    # t = Estimate / Std. Error
    finite <- complete.cases(cmat[, c("Estimate", "Std. Error", "t value")])
    if (any(finite)){
        expect_equal(cmat[finite, "t value"],
                     cmat[finite, "Estimate"] / cmat[finite, "Std. Error"],
                     tolerance = 1e-10)
    }
})

#### accuracy + pls ####
test_that("accuracy.pts works with object$holdout", {
    a <- accuracy(m)
    expect_true(length(a) > 0)
    expect_true("MAE" %in% names(a))
})

test_that("pls.pts is the sum of squared forecast errors", {
    p <- pls(m)
    h <- length(m$holdout)
    fhat <- as.numeric(forecast(m, h = h)$mean)
    expect_equal(p, sum((as.numeric(m$holdout) - fhat)^2), tolerance = 1e-10)
})

#### update ####
test_that("update.pts re-runs pts with the new args", {
    m2 <- update(m, model = "1NT")
    expect_s3_class(m2, "pts")
    expect_equal(m2$lambda, 1)
})

#### simulate ####
test_that("simulate.pts returns an h x nsim path matrix", {
    s <- simulate(m, nsim = 50, h = 12, seed = 1L)
    expect_s3_class(s, "pts.sim")
    expect_equal(dim(s$data), c(12L, 50L))
})

test_that("simulate.pts is reproducible with a seed", {
    s1 <- simulate(m, nsim = 50, h = 12, seed = 7L)
    s2 <- simulate(m, nsim = 50, h = 12, seed = 7L)
    expect_identical(s1$data, s2$data)
})

test_that("simulate.pts sample-path mean -> forecast mean", {
    # 2000 paths should bring the mean close to the deterministic forecast.
    s <- simulate(m, nsim = 2000, h = 12, seed = 11L)
    fmean <- as.numeric(forecast(m, h = 12)$mean)
    expect_lt(max(abs(rowMeans(s$data) - fmean)), 0.01)
})

#### adam-aligned slots ####
test_that("pts carries the adam slots that plot.smooth reads", {
    expect_equal(m$distribution, "dnorm")
    expect_equal(m$loss, "likelihood")
    expect_null(m$occurrence)
    expect_null(m$persistence)
    expect_null(m$phi)
    expect_null(m$transition)
    # states: structural columns of comp (no Error / Fit), in-sample only.
    expect_true(is.matrix(m$states))
    expect_equal(nrow(m$states), length(m$y))
    expect_false("Error" %in% colnames(m$states))
    expect_false("Fit"   %in% colnames(m$states))
})

#### plot inheritance ####
test_that("plot(m) dispatches to plot.smooth without error", {
    pdf(NULL)
    on.exit(dev.off())
    expect_silent(plot(m))
})

test_that("every plot.smooth panel (which = 1..16) runs without error", {
    pdf(NULL)
    on.exit(dev.off())
    for (w in 1:16) {
        expect_silent(plot(m, which = w))
    }
})

test_that("plot(forecast(m)) dispatches to plot.smooth.forecast", {
    pdf(NULL)
    on.exit(dev.off())
    f <- forecast(m, h = 12)
    expect_silent(plot(f))
})
