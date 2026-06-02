#### PTSsetup ####
test_that("PTSsetup returns a PTS object with default ZZZ model", {
    m <- PTSsetup(log(AirPassengers))
    expect_s3_class(m, "PTS")
    expect_equal(m$model, "ZZZ")
    expect_equal(m$h, 12)
    expect_equal(m$criterion, "aic")
    expect_false(m$armaIdent)
    expect_false(m$verbose)
    expect_equal(m$s, frequency(AirPassengers))
    expect_equal(m$modelUC, "?/none/?/arma(0,0)")
    expect_equal(m$lambda, 9999.9)
})

test_that("PTSsetup preserves user-supplied arguments", {
    y <- log(AirPassengers)
    m <- PTSsetup(y, model = "0NT", h = 24, criterion = "bic",
                  armaIdent = TRUE, verbose = TRUE)
    expect_equal(m$model, "0NT")
    expect_equal(m$h, 24)
    expect_equal(m$criterion, "bic")
    expect_true(m$armaIdent)
    expect_true(m$verbose)
    expect_equal(m$lambda, 0)
    expect_equal(m$modelUC, "rw/none/equal/arma(0,0)")
})

test_that("PTSsetup initialises output slots to NA", {
    m <- PTSsetup(log(AirPassengers))
    expect_true(all(is.na(m$p0)))
    expect_true(all(is.na(m$p)))
    expect_true(all(is.na(m$v)))
    expect_true(all(is.na(m$yFor)))
    expect_true(all(is.na(m$yForV)))
    expect_true(all(is.na(m$comp)))
    expect_identical(m$table, "")
    expect_equal(m$armaOrders, c(0, 0))
})

test_that("PTSsetup accepts a numeric vector with explicit s", {
    set.seed(42)
    y <- as.numeric(log(AirPassengers))
    m <- PTSsetup(y, model = "1NT", s = 12, h = 12)
    expect_s3_class(m, "PTS")
    expect_equal(m$lambda, 1)
    expect_equal(m$modelUC, "rw/none/equal/arma(0,0)")
})

#### PTS2modelUC ####
test_that("PTS2modelUC translates ZZZ to fully-automatic UC model", {
    out <- PTS2modelUC("ZZZ")
    expect_equal(out$modelU, "?/none/?/arma(0,0)")
    expect_equal(out$lambda, 9999.9)
})

test_that("PTS2modelUC translates trend letters", {
    expect_equal(PTS2modelUC("1NN")$modelU, "rw/none/none/arma(0,0)")
    expect_equal(PTS2modelUC("1LN")$modelU, "llt/none/none/arma(0,0)")
    expect_equal(PTS2modelUC("1DN")$modelU, "srw/none/none/arma(0,0)")
    expect_equal(PTS2modelUC("1GN")$modelU, "td/none/none/arma(0,0)")
})

test_that("PTS2modelUC translates seasonal letters", {
    expect_equal(PTS2modelUC("1NN")$modelU, "rw/none/none/arma(0,0)")
    expect_equal(PTS2modelUC("1ND")$modelU, "rw/none/linear/arma(0,0)")
    expect_equal(PTS2modelUC("1NT")$modelU, "rw/none/equal/arma(0,0)")
})

test_that("PTS2modelUC parses numeric power into lambda", {
    expect_equal(PTS2modelUC("0NN")$lambda, 0)
    expect_equal(PTS2modelUC("1NN")$lambda, 1)
    expect_equal(PTS2modelUC("0.5NN")$lambda, 0.5)
    expect_equal(PTS2modelUC("-1NN")$lambda, -1)
})

test_that("PTS2modelUC carries through arma orders", {
    out <- PTS2modelUC("1NT", armaOrders = c(2, 1))
    expect_equal(out$modelU, "rw/none/equal/arma(2,1)")
})

test_that("PTS2modelUC rejects invalid component letters", {
    expect_error(PTS2modelUC("1XT"), "incorrect trend model")
    expect_error(PTS2modelUC("1NX"), "incorrect seasonal model")
    expect_error(PTS2modelUC("XNN"), "incorrect power model")
})

#### modelUC2PTS ####
test_that("modelUC2PTS reverses PTS2modelUC", {
    expect_equal(modelUC2PTS("rw/none/none/arma(0,0)",  1), "1NN")
    expect_equal(modelUC2PTS("llt/none/equal/arma(0,0)", 0), "0LT")
    expect_equal(modelUC2PTS("td/none/linear/arma(0,0)", 0.5), "0.5GD")
    expect_equal(modelUC2PTS("srw/none/none/arma(0,0)", -1), "-1DN")
})

#### modelUC2arma ####
test_that("modelUC2arma extracts ARMA orders", {
    expect_equal(modelUC2arma("rw/none/none/arma(0,0)"), c(0, 0))
    expect_equal(modelUC2arma("rw/none/equal/arma(2,1)"), c(2, 1))
    expect_equal(modelUC2arma("llt/none/equal/arma(3,2)"), c(3, 2))
})
