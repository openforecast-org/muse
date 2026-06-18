## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 4
)

## ----setup--------------------------------------------------------------------
library(muse)

## ----air-default--------------------------------------------------------------
air <- pts(AirPassengers, model = "ZZZ", h = 12, holdout = TRUE)
air

## ----air-fit, fig.height=6----------------------------------------------------
plot(air, which = c(1, 2))

## ----air-states, fig.height=6-------------------------------------------------
plot(air, which = 12)

## ----air-fc-------------------------------------------------------------------
fc <- forecast(air, h = 12)
plot(fc)

## ----air-acc------------------------------------------------------------------
accuracy(air)

## ----air-loglambda------------------------------------------------------------
air_log <- pts(AirPassengers, model = "0LT", h = 12, holdout = TRUE)
air_log$lambda

## ----air-auto, eval = FALSE---------------------------------------------------
# pts(AirPassengers, model = "ZZZ", h = 12, ic = "BIC")

## ----air-arma, eval = FALSE---------------------------------------------------
# # ARMA(2, 1) on the irregular
# pts(AirPassengers, model = "ZZZ", orders = c(2, 1), h = 12)
# 
# # Seasonal SARMA(1, 1)(1, 1)_12 — non-seasonal + seasonal blocks
# pts(AirPassengers, model = "ZZZ",
#     orders = list(ar = c(1, 1), ma = c(1, 1), lags = c(1, 12)), h = 12)
# 
# # Automatic ARMA search up to the supplied caps
# pts(AirPassengers, model = "ZZZ",
#     orders = list(ar = 2, ma = 2, select = TRUE), h = 12)

## ----air-fc-intervals---------------------------------------------------------
fc <- forecast(air, h = 12, interval = "prediction", level = c(0.80, 0.95))
plot(fc)

## ----air-outliers-------------------------------------------------------------
y <- AirPassengers
y[100] <- 3 * y[100]            # inject an obvious additive spike

m_out <- pts(y, model = "ZZZ", outliers = "use", level = 0.99)
m_out$outliersDetected

## ----seatbelts-xreg-----------------------------------------------------------
sb <- Seatbelts[, c("drivers", "kms", "PetrolPrice", "law")]
m_sb <- pts(sb, model = "ZZZ", h = 12, holdout = TRUE)
m_sb

## ----seatbelts-fc, eval = FALSE-----------------------------------------------
# # Holdout values of the regressors are stashed on $holdout for accuracy
# fc_sb <- forecast(m_sb, h = 12, newdata = tail(Seatbelts, 12))
# plot(fc_sb)

## ----seatbelts-outliers-------------------------------------------------------
m_sb_out <- pts(Seatbelts[, "drivers"], model = "ZZZ",
                outliers = "use", level = 0.99)
m_sb_out$outliersDetected

## ----air-accuracy-------------------------------------------------------------
accuracy(air)

## ----air-sim------------------------------------------------------------------
set.seed(42)
sim <- simulate(air, nsim = 200, h = 12)
str(sim, max.level = 1)

## ----air-accessors, eval = FALSE----------------------------------------------
# summary(air)         # Coefficient table + variance proportions
# coef(air)            # Estimated parameter vector
# vcov(air)            # Parameter covariance matrix
# confint(air)         # Wald confidence intervals
# logLik(air); AIC(air); BIC(air)
# fitted(air); residuals(air)
# rstandard(air); rstudent(air)
# nparam(air); nobs(air); sigma(air)
# modelType(air); orders(air); lags(air); errorType(air)

