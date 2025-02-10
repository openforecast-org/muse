context("Tests for CTLL")

#### Basic CTLL stuff ####
testModel <- ctll(BJsales, h=10, holdout=TRUE)
testForecast <- forecast(testModel, h=10)
test_that("CTLL on BJsales", {
    expect_true(all(testModel$forecast %in% testForecast$mean))
})
